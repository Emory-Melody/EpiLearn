import torch
import numpy as np
from collections.abc import Mapping, Sequence as SequenceCollection

_NUMBER_TYPES = (int, float, np.floating, np.integer)

def _ensure_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    if isinstance(value, _NUMBER_TYPES):
        return float(value)
    raise TypeError(f"Unsupported parameter value type: {type(value)}")


def _resolve_schedule(schedule, step_idx, t, state):
    if schedule is None:
        return None
    if callable(schedule):
        return schedule(step_idx, t, state)
    if isinstance(schedule, Mapping):
        value = schedule.get(step_idx)
        if callable(value):
            return value(step_idx, t, state)
        return value
    if isinstance(schedule, SequenceCollection) and not isinstance(schedule, (str, bytes)):
        if step_idx < len(schedule):
            value = schedule[step_idx]
            if callable(value):
                return value(step_idx, t, state)
            return value
        return None
    return schedule


class CompartmentalModel:
    """
    Base class for deterministic compartmental epidemic models.
    """

    def __init__(self, compartments, parameters, metadata=None, name=None):
        if not compartments:
            raise ValueError("At least one compartment is required.")
        self.compartments = tuple(compartments)
        self.parameters = {key: _ensure_float(value) for key, value in parameters.items()}
        self.metadata = metadata or {}
        self.name = name or self.__class__.__name__

    def rhs(self, state, t, external_inputs=None, parameters=None):
        raise NotImplementedError("Subclasses must implement the RHS of the dynamical system.")

    def validate_state(self, state):
        tensor = torch.as_tensor(state, dtype=torch.float32)
        if tensor.shape[-1] != len(self.compartments):
            raise ValueError(f"State dimension {tensor.shape} does not match compartments {self.compartments}.")
        return tensor

    def project_state(self, state):
        return torch.clamp(state, min=0.0)

    def _merge_parameters(self, overrides=None):
        params = dict(self.parameters)
        if overrides:
            params.update({k: _ensure_float(v) for k, v in overrides.items()})
        return params

    def step(self, state, t, dt, method='rk4', external_inputs=None, parameter_overrides=None):
        state = self.validate_state(state)
        params = self._merge_parameters(parameter_overrides)
        if method not in {'rk4', 'euler'}:
            raise ValueError("Supported integration methods are 'rk4' and 'euler'.")
        if method == 'euler':
            delta = self.rhs(state, t, external_inputs=external_inputs, parameters=params)
            next_state = state + dt * delta
        else:
            k1 = self.rhs(state, t, external_inputs=external_inputs, parameters=params)
            k2 = self.rhs(state + 0.5 * dt * k1, t + 0.5 * dt, external_inputs=external_inputs, parameters=params)
            k3 = self.rhs(state + 0.5 * dt * k2, t + 0.5 * dt, external_inputs=external_inputs, parameters=params)
            k4 = self.rhs(state + dt * k3, t + dt, external_inputs=external_inputs, parameters=params)
            next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self.project_state(next_state)

    def simulate(self, initial_state, steps, dt=1.0, method='rk4', parameter_schedule=None, input_schedule=None):
        state = self.validate_state(initial_state)
        history = torch.zeros(steps + 1, len(self.compartments), dtype=state.dtype)
        history[0] = state
        for step_idx in range(steps):
            t = step_idx * dt
            overrides = _resolve_schedule(parameter_schedule, step_idx, t, state)
            inputs = _resolve_schedule(input_schedule, step_idx, t, state)
            state = self.step(state, t, dt, method=method, external_inputs=inputs, parameter_overrides=overrides)
            history[step_idx + 1] = state
        time_axis = torch.linspace(0.0, steps * dt, steps + 1)
        return {"time": time_axis, "trajectory": history, "compartments": self.compartments}


class SIRModel(CompartmentalModel):
    """
    Classical SIR model with optional demography.
    """

    def __init__(self, beta, gamma, mu=0.0, birth_rate=0.0):
        metadata = {'infectious_compartments': ('I',), 'recovery_target': 'R'}
        params = {'beta': beta, 'gamma': gamma, 'mu': mu, 'birth_rate': birth_rate}
        super().__init__(('S', 'I', 'R'), params, metadata=metadata, name="SIR")

    def rhs(self, state, t, external_inputs=None, parameters=None):
        params = parameters or self.parameters
        beta = params['beta']
        gamma = params['gamma']
        mu = params.get('mu', 0.0)
        birth_rate = params.get('birth_rate', 0.0)
        S, I, R = state
        N = torch.clamp(state.sum(), min=1e-8)
        force = beta * I / N
        if external_inputs and 'force_of_infection' in external_inputs:
            force = external_inputs['force_of_infection']
        births = birth_rate * N
        dS = births - force * S - mu * S
        dI = force * S - (gamma + mu) * I
        dR = gamma * I - mu * R
        return torch.stack([dS, dI, dR])


class SEIRModel(CompartmentalModel):
    """
    SEIR model (Susceptible-Exposed-Infectious-Recovered).
    """

    def __init__(self, beta, gamma, sigma, mu=0.0, birth_rate=0.0):
        metadata = {'infectious_compartments': ('I',), 'recovery_target': 'R'}
        params = {'beta': beta, 'gamma': gamma, 'sigma': sigma, 'mu': mu, 'birth_rate': birth_rate}
        super().__init__(('S', 'E', 'I', 'R'), params, metadata=metadata, name="SEIR")

    def rhs(self, state, t, external_inputs=None, parameters=None):
        params = parameters or self.parameters
        beta = params['beta']
        gamma = params['gamma']
        sigma = params['sigma']
        mu = params.get('mu', 0.0)
        birth_rate = params.get('birth_rate', 0.0)
        S, E, I, R = state
        N = torch.clamp(state.sum(), min=1e-8)
        force = beta * I / N
        if external_inputs and 'force_of_infection' in external_inputs:
            force = external_inputs['force_of_infection']
        births = birth_rate * N
        dS = births - force * S - mu * S
        dE = force * S - (sigma + mu) * E
        dI = sigma * E - (gamma + mu) * I
        dR = gamma * I - mu * R
        return torch.stack([dS, dE, dI, dR])


class SIRSModel(CompartmentalModel):
    """
    SIRS model with waning immunity from R back to S.
    """

    def __init__(self, beta, gamma, omega, mu=0.0, birth_rate=0.0):
        metadata = {
            'infectious_compartments': ('I',),
            'recovery_target': 'R',
            'waning_target': 'S'
        }
        params = {'beta': beta, 'gamma': gamma, 'omega': omega, 'mu': mu, 'birth_rate': birth_rate}
        super().__init__(('S', 'I', 'R'), params, metadata=metadata, name="SIRS")

    def rhs(self, state, t, external_inputs=None, parameters=None):
        params = parameters or self.parameters
        beta = params['beta']
        gamma = params['gamma']
        omega = params['omega']
        mu = params.get('mu', 0.0)
        birth_rate = params.get('birth_rate', 0.0)
        S, I, R = state
        N = torch.clamp(state.sum(), min=1e-8)
        force = beta * I / N
        if external_inputs and 'force_of_infection' in external_inputs:
            force = external_inputs['force_of_infection']
        births = birth_rate * N
        dS = births - force * S - mu * S + omega * R
        dI = force * S - (gamma + mu) * I
        dR = gamma * I - (mu + omega) * R
        return torch.stack([dS, dI, dR])


class SEIRVIModel(CompartmentalModel):
    """
    SEIR model with Vaccination and Isolation interventions.
    
    Compartments: S (Susceptible), E (Exposed), I (Infectious), R (Recovered), V (Vaccinated), Q (Isolated/Quarantined)
    
    Intervention effects:
    - Vaccination: Moves susceptibles to V with rate dependent on vaccination policy
    - Isolation: Moves infectious individuals to Q, reducing transmission
    
    Parameters
    ----------
    beta : float
        Base transmission rate.
    gamma : float
        Recovery rate (1/infectious_period).
    sigma : float
        Incubation rate (1/latent_period).
    mu : float
        Natural death rate (default 0).
    birth_rate : float
        Birth rate (default 0).
    vaccine_efficacy : float
        Vaccine efficacy in preventing infection (0-1, default 0.8).
    isolation_efficacy : float
        Reduction in transmission from isolated individuals (0-1, default 0.9).
    
    Intervention inputs (via external_inputs or schedules):
    - vaccination_rate: Rate of vaccination (fraction per timestep)
    - vaccination_delay: Delay before vaccination takes effect (in timesteps)
    - isolation_rate: Rate of isolating infectious individuals
    - isolation_delay: Delay before isolation policy takes effect
    """
    
    def __init__(self, beta, gamma, sigma, mu=0.0, birth_rate=0.0, 
                 vaccine_efficacy=0.8, isolation_efficacy=0.9):
        metadata = {
            'infectious_compartments': ('I',),
            'recovery_target': 'R',
            'intervention_compartments': ('V', 'Q'),
        }
        params = {
            'beta': beta, 'gamma': gamma, 'sigma': sigma, 
            'mu': mu, 'birth_rate': birth_rate,
            'vaccine_efficacy': vaccine_efficacy,
            'isolation_efficacy': isolation_efficacy,
        }
        # Compartments: S, E, I, R, V (Vaccinated), Q (Isolated/Quarantined)
        super().__init__(('S', 'E', 'I', 'R', 'V', 'Q'), params, metadata=metadata, name="SEIR-VI")

    def rhs(self, state, t, external_inputs=None, parameters=None):
        params = parameters or self.parameters
        beta = params['beta']
        gamma = params['gamma']
        sigma = params['sigma']
        mu = params.get('mu', 0.0)
        birth_rate = params.get('birth_rate', 0.0)
        vaccine_efficacy = params.get('vaccine_efficacy', 0.8)
        isolation_efficacy = params.get('isolation_efficacy', 0.9)
        
        S, E, I, R, V, Q = state
        N = torch.clamp(state.sum(), min=1e-8)
        
        # Effective infectious population (isolated individuals transmit less)
        I_effective = I + (1 - isolation_efficacy) * Q
        
        # Force of infection
        force = beta * I_effective / N
        if external_inputs and 'force_of_infection' in external_inputs:
            force = external_inputs['force_of_infection']
        
        # Intervention rates from external inputs
        vaccination_rate = 0.0
        isolation_rate = 0.0
        
        if external_inputs:
            vaccination_rate = external_inputs.get('vaccination_rate', 0.0)
            isolation_rate = external_inputs.get('isolation_rate', 0.0)
        
        births = birth_rate * N
        
        # Dynamics with interventions
        # S -> V (vaccination), S -> E (infection)
        dS = births - force * S - mu * S - vaccination_rate * S
        
        # E -> I (progression)
        dE = force * S + (1 - vaccine_efficacy) * force * V - (sigma + mu) * E
        
        # I -> R (recovery), I -> Q (isolation)
        dI = sigma * E - (gamma + mu) * I - isolation_rate * I
        
        # R from I and Q
        dR = gamma * I + gamma * Q - mu * R
        
        # V: vaccinated (can still get infected with reduced probability)
        dV = vaccination_rate * S - (1 - vaccine_efficacy) * force * V - mu * V
        
        # Q: isolated/quarantined (from I, recover at same rate)
        dQ = isolation_rate * I - gamma * Q - mu * Q
        
        return torch.stack([dS, dE, dI, dR, dV, dQ])
    
    def generate_scenario_samples(
        self,
        initial_state,
        lookback: int,
        horizon: int,
        intervention_trajectories: torch.Tensor,
        dt: float = 1.0,
        method: str = 'rk4',
        process_noise: float = None,
        seed: int = None,
    ):
        """
        Generate simulation samples for scenario modeling.
        
        This function generates both historical trajectories (with interventions applied
        according to their delays) and future projections.
        
        Parameters
        ----------
        initial_state : array-like
            Initial state for compartments [S, E, I, R, V, Q].
        lookback : int
            Lookback window size L (historical period).
        horizon : int
            Horizon size H (future projection period).
        intervention_trajectories : torch.Tensor
            Intervention trajectories of shape (L+H, 2, 2):
            - First dimension: timesteps
            - Second dimension: [vaccination, isolation]
            - Third dimension: [value, delay]
            Example: intervention_trajectories[t, 0, :] = [vaccination_rate, vaccination_delay]
                     intervention_trajectories[t, 1, :] = [isolation_rate, isolation_delay]
        dt : float
            Time step size (default 1.0).
        method : str
            Integration method ('rk4' or 'euler').
        process_noise : float, optional
            Standard deviation of process noise.
        seed : int, optional
            Random seed for reproducibility.
        
        Returns
        -------
        dict
            - 'historical': Tensor of shape (L, N+4) where N is number of compartments (6),
              and +4 is for [vaccination_value, vaccination_delay, isolation_value, isolation_delay]
            - 'future': Tensor of shape (H, N) containing future compartment trajectories
            - 'time': Time axis for entire simulation
            - 'compartments': Compartment names
        """
        state = self.validate_state(initial_state)
        total_steps = lookback + horizon
        N = len(self.compartments)  # 6 compartments
        
        # Output tensors
        historical = torch.zeros(lookback, N + 4, dtype=state.dtype)
        future = torch.zeros(horizon, N, dtype=state.dtype)
        
        # Set up noise
        rng = None
        noise_std = None
        if process_noise is not None:
            rng = torch.Generator()
            if seed is not None:
                rng.manual_seed(seed)
            noise_std = torch.full((N,), process_noise, dtype=state.dtype)
        
        # Simulate
        current_state = state.clone()
        
        for step_idx in range(total_steps):
            t = step_idx * dt
            
            # Get intervention values and delays for current step
            vacc_value = float(intervention_trajectories[step_idx, 0, 0])
            vacc_delay = int(intervention_trajectories[step_idx, 0, 1])
            isol_value = float(intervention_trajectories[step_idx, 1, 0])
            isol_delay = int(intervention_trajectories[step_idx, 1, 1])
            
            # Apply interventions with delay
            # Intervention takes effect only if current step >= delay
            effective_vacc = vacc_value if step_idx >= vacc_delay else 0.0
            effective_isol = isol_value if step_idx >= isol_delay else 0.0
            
            external_inputs = {
                'vaccination_rate': effective_vacc,
                'isolation_rate': effective_isol,
            }
            
            # Record state
            if step_idx < lookback:
                # Historical: compartments + intervention info
                historical[step_idx, :N] = current_state
                historical[step_idx, N] = vacc_value
                historical[step_idx, N+1] = vacc_delay
                historical[step_idx, N+2] = isol_value
                historical[step_idx, N+3] = isol_delay
            else:
                # Future: only compartments
                future[step_idx - lookback, :] = current_state
            
            # Step the model
            next_state = self.step(current_state, t, dt, method=method, 
                                   external_inputs=external_inputs)
            
            # Add process noise
            if noise_std is not None and rng is not None:
                noise = torch.randn(N, generator=rng) * noise_std
                next_state = self.project_state(next_state + noise)
            
            current_state = next_state
        
        time_axis = torch.linspace(0.0, total_steps * dt, total_steps)
        
        return {
            'historical': historical,
            'future': future,
            'time': time_axis,
            'compartments': self.compartments,
            'lookback': lookback,
            'horizon': horizon,
        }

    def generate_multi_scenario_samples(
        self,
        initial_state,
        lookback: int,
        horizon: int,
        intervention_scenarios: torch.Tensor,
        dt: float = 1.0,
        method: str = 'rk4',
        process_noise: float = None,
        seed: int = None,
    ):
        """
        Generate simulation samples for multiple scenarios with shared history.
        
        This function generates:
        - A single historical trajectory that is SHARED across all scenarios
        - Multiple future projections, one per scenario
        
        Key concept:
        - **Static interventions**: t + delay < L (take effect within historical period)
          Must be IDENTICAL across all scenarios.
        - **Dynamic interventions**: t + delay >= L (take effect in horizon period)
          Can VARY across scenarios.
        
        Parameters
        ----------
        initial_state : array-like
            Initial state for compartments [S, E, I, R, V, Q].
        lookback : int
            Lookback window size L (historical period).
        horizon : int
            Horizon size H (future projection period).
        intervention_scenarios : torch.Tensor
            Intervention scenarios of shape (L+H, n_scenarios, 2, 2):
            - Dim 0: timesteps (L+H total)
            - Dim 1: scenarios
            - Dim 2: [vaccination, isolation]
            - Dim 3: [value, delay]
            
            Static entries (where t + delay < L) MUST be identical across scenarios.
            Only dynamic entries (where t + delay >= L) can differ.
        dt : float
            Time step size (default 1.0).
        method : str
            Integration method ('rk4' or 'euler').
        process_noise : float, optional
            Standard deviation of process noise (applied identically across scenarios
            during historical period, independently during horizon).
        seed : int, optional
            Random seed for reproducibility.
        
        Returns
        -------
        dict
            - 'historical': Tensor of shape (L, n_scenarios, N+4)
              Compartment values (N) are IDENTICAL across scenarios.
              Intervention columns (+4) may differ for dynamic interventions.
            - 'future': Tensor of shape (H, n_scenarios, N)
              Different future trajectories per scenario.
            - 'time': Time axis for entire simulation
            - 'compartments': Compartment names
            - 'n_scenarios': Number of scenarios
            - 'static_mask': Boolean mask indicating static intervention entries
            - 'dynamic_mask': Boolean mask indicating dynamic intervention entries
        """
        total_steps = lookback + horizon
        N = len(self.compartments)
        
        # Validate input shape
        assert intervention_scenarios.dim() == 4, \
            f"intervention_scenarios must be 4D (T, scenarios, 2, 2), got {intervention_scenarios.dim()}D"
        assert intervention_scenarios.shape[0] == total_steps, \
            f"First dim must be L+H={total_steps}, got {intervention_scenarios.shape[0]}"
        assert intervention_scenarios.shape[2:] == (2, 2), \
            f"Last dims must be (2, 2), got {intervention_scenarios.shape[2:]}"
        
        n_scenarios = intervention_scenarios.shape[1]
        
        # Build static/dynamic masks and validate static entries are identical
        static_mask = torch.zeros(total_steps, 2, dtype=torch.bool)
        dynamic_mask = torch.zeros(total_steps, 2, dtype=torch.bool)
        
        for t in range(total_steps):
            for interv_idx in range(2):
                delays = intervention_scenarios[t, :, interv_idx, 1]
                values = intervention_scenarios[t, :, interv_idx, 0]
                is_static = torch.all(t + delays < lookback)
                
                if is_static:
                    static_mask[t, interv_idx] = True
                    assert torch.allclose(values, values[0].expand_as(values)), \
                        f"Static intervention at t={t}, interv={interv_idx}: values must match. Got {values.tolist()}"
                    assert torch.allclose(delays, delays[0].expand_as(delays)), \
                        f"Static intervention at t={t}, interv={interv_idx}: delays must match. Got {delays.tolist()}"
                else:
                    dynamic_mask[t, interv_idx] = True
        assert dynamic_mask.sum() != 0 , "At least one dynamic intervention entry is required."
        # Use generate_scenario_samples for first scenario to get shared historical trajectory
        base_result = self.generate_scenario_samples(
            initial_state=initial_state,
            lookback=lookback,
            horizon=horizon,
            intervention_trajectories=intervention_scenarios[:, 0, :, :],  # First scenario
            dt=dt,
            method=method,
            process_noise=process_noise,
            seed=seed,
        )
        
        # Build historical tensor: expand shared compartments, add scenario-specific interventions
        shared_hist = base_result['historical'][:, :N]  # (L, N) - shared compartments
        historical = torch.zeros(lookback, n_scenarios, N + 4, dtype=shared_hist.dtype)
        
        for s in range(n_scenarios):
            historical[:, s, :N] = shared_hist  # Same compartments for all
            historical[:, s, N:] = intervention_scenarios[:lookback, s, :, :].reshape(lookback, 4)
        
        # Get final historical state to start horizon simulations
        final_state = shared_hist[-1].clone()
        # Step one more time to get state at t=lookback
        vacc_val = float(intervention_scenarios[lookback - 1, 0, 0, 0])
        vacc_del = int(intervention_scenarios[lookback - 1, 0, 0, 1])
        isol_val = float(intervention_scenarios[lookback - 1, 0, 1, 0])
        isol_del = int(intervention_scenarios[lookback - 1, 0, 1, 1])
        external = {
            'vaccination_rate': vacc_val if (lookback - 1) >= vacc_del else 0.0,
            'isolation_rate': isol_val if (lookback - 1) >= isol_del else 0.0,
        }
        final_state = self.step(final_state, (lookback - 1) * dt, dt, method=method, external_inputs=external)
        
        # Generate future for each scenario independently
        future = torch.zeros(horizon, n_scenarios, N, dtype=shared_hist.dtype)
        
        # Set up noise for horizon period
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed + 1000)  # Different seed from historical
        noise_std = torch.full((N,), process_noise, dtype=shared_hist.dtype) if process_noise else None
        
        for s in range(n_scenarios):
            scenario_state = final_state.clone()
            
            for h in range(horizon):
                future[h, s, :] = scenario_state
                step_idx = lookback + h
                
                vacc_value = float(intervention_scenarios[step_idx, s, 0, 0])
                vacc_delay = int(intervention_scenarios[step_idx, s, 0, 1])
                isol_value = float(intervention_scenarios[step_idx, s, 1, 0])
                isol_delay = int(intervention_scenarios[step_idx, s, 1, 1])
                
                external_inputs = {
                    'vaccination_rate': vacc_value if step_idx >= vacc_delay else 0.0,
                    'isolation_rate': isol_value if step_idx >= isol_delay else 0.0,
                }
                
                scenario_state = self.step(scenario_state, step_idx * dt, dt, method=method,
                                           external_inputs=external_inputs)
                
                if noise_std is not None:
                    noise = torch.randn(N, generator=rng) * noise_std
                    scenario_state = self.project_state(scenario_state + noise)
        
        return {
            'historical': historical,
            'future': future,
            'time': base_result['time'],
            'compartments': self.compartments,
            'lookback': lookback,
            'horizon': horizon,
            'n_scenarios': n_scenarios,
            'static_mask': static_mask,
            'dynamic_mask': dynamic_mask,
        }

    def create_multi_scenario_interventions(
        self,
        lookback: int,
        horizon: int,
        n_scenarios: int,
        static_vacc_rate: float,
        static_vacc_delay: int,
        static_isol_rate: float,
        static_isol_delay: int,
        dynamic_vacc_rates: list,
        dynamic_vacc_delays: list,
        dynamic_isol_rates: list,
        dynamic_isol_delays: list,
    ):
        """
        Create intervention scenarios tensor ensuring static/dynamic constraints.
        
        Static interventions (t + delay < L) are applied identically.
        Dynamic interventions (t + delay >= L) vary per scenario.
        
        Returns:
            intervention_scenarios: Tensor (L+H, n_scenarios, 2, 2)
        """
        total_steps = lookback + horizon
        scenarios = torch.zeros(total_steps, n_scenarios, 2, 2)
        
        for t in range(total_steps):
            for s in range(n_scenarios):
                # Vaccination
                vacc_delay = static_vacc_delay if t + static_vacc_delay < lookback else dynamic_vacc_delays[s]
                vacc_rate = static_vacc_rate if t + static_vacc_delay < lookback else dynamic_vacc_rates[s]
                scenarios[t, s, 0, 0] = vacc_rate
                scenarios[t, s, 0, 1] = vacc_delay
                
                # Isolation
                isol_delay = static_isol_delay if t + static_isol_delay < lookback else dynamic_isol_delays[s]
                isol_rate = static_isol_rate if t + static_isol_delay < lookback else dynamic_isol_rates[s]
                scenarios[t, s, 1, 0] = isol_rate
                scenarios[t, s, 1, 1] = isol_delay
        
        return scenarios

    def generate_multi_scenario_dataset(
        self,
        n_samples: int,
        lookback: int,
        horizon: int,
        n_scenarios: int,
        static_vacc_rate_range: tuple = (0.0, 0.02),
        static_vacc_delay_range: tuple = (10, 30),
        static_isol_rate_range: tuple = (0.0, 0.1),
        static_isol_delay_range: tuple = (5, 20),
        dynamic_vacc_rate_range: tuple = (0.0, 0.08),
        dynamic_isol_rate_range: tuple = (0.0, 0.3),
        initial_infected_frac_range: tuple = (0.001, 0.02),
        population: float = 1e6,
        process_noise: float = None,
        seed: int = 42,
    ):
        """
        Generate a dataset using generate_multi_scenario_samples.
        
        Each sample generates:
        - X: (L, n_scenarios, N+4) - shared history with scenario-specific interventions
        - Y: (H, n_scenarios, N) - different futures per scenario
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        lookback : int
            Historical window size L.
        horizon : int
            Future projection size H.
        n_scenarios : int
            Number of scenarios per sample.
        static_vacc_rate_range : tuple
            Range for static vaccination rate (min, max).
        static_vacc_delay_range : tuple
            Range for static vaccination delay (min, max).
        static_isol_rate_range : tuple
            Range for static isolation rate (min, max).
        static_isol_delay_range : tuple
            Range for static isolation delay (min, max).
        dynamic_vacc_rate_range : tuple
            Range for dynamic vaccination rate (min, max).
        dynamic_isol_rate_range : tuple
            Range for dynamic isolation rate (min, max).
        initial_infected_frac_range : tuple
            Range for initial infected fraction (min, max).
        population : float
            Total population size.
        process_noise : float, optional
            Standard deviation of process noise. If None, no noise is added.
            Noise is applied identically during historical period and independently
            per scenario during horizon period.
        seed : int
            Random seed for reproducibility.
        
        Returns
        -------
        X : torch.Tensor
            Shape (n_samples, L, n_scenarios, N+4)
        Y : torch.Tensor
            Shape (n_samples, H, n_scenarios, N)
        metadata : list
            List of sample metadata dictionaries.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        N = len(self.compartments)
        
        X_all = torch.zeros(n_samples, lookback, n_scenarios, N + 4)
        Y_all = torch.zeros(n_samples, horizon, n_scenarios, N)
        metadata_all = []
        
        for i in range(n_samples):
            # Random static intervention parameters (same across scenarios)
            static_vacc_rate = np.random.uniform(*static_vacc_rate_range)
            static_vacc_delay = np.random.randint(*static_vacc_delay_range)
            static_isol_rate = np.random.uniform(*static_isol_rate_range)
            static_isol_delay = np.random.randint(*static_isol_delay_range)
            
            # Ensure delays make interventions static (t + delay < lookback for early t)
            # Adjust delays to be within valid static range
            static_vacc_delay = min(static_vacc_delay, lookback - 10)
            static_isol_delay = min(static_isol_delay, lookback - 10)
            
            # Random dynamic intervention parameters (vary per scenario)
            dynamic_vacc_rates = np.random.uniform(*dynamic_vacc_rate_range, size=n_scenarios).tolist()
            dynamic_isol_rates = np.random.uniform(*dynamic_isol_rate_range, size=n_scenarios).tolist()
            # Dynamic delays at or after lookback
            dynamic_vacc_delays = [lookback] * n_scenarios
            dynamic_isol_delays = [lookback] * n_scenarios
            
            # Random initial conditions
            init_infected = np.random.uniform(*initial_infected_frac_range)
            initial_state = torch.tensor([
                population * (1 - init_infected),  # S
                population * init_infected,         # E
                0, 0, 0, 0                          # I, R, V, Q
            ], dtype=torch.float32)
            
            # Create multi-scenario interventions
            intervention_scenarios = self.create_multi_scenario_interventions(
                lookback=lookback,
                horizon=horizon,
                n_scenarios=n_scenarios,
                static_vacc_rate=static_vacc_rate,
                static_vacc_delay=static_vacc_delay,
                static_isol_rate=static_isol_rate,
                static_isol_delay=static_isol_delay,
                dynamic_vacc_rates=dynamic_vacc_rates,
                dynamic_vacc_delays=dynamic_vacc_delays,
                dynamic_isol_rates=dynamic_isol_rates,
                dynamic_isol_delays=dynamic_isol_delays,
            )
            
            # Generate multi-scenario sample with noise
            result = self.generate_multi_scenario_samples(
                initial_state=initial_state,
                lookback=lookback,
                horizon=horizon,
                intervention_scenarios=intervention_scenarios,
                dt=1.0,
                method='rk4',
                process_noise=process_noise,
                seed=seed + i if seed is not None else None,
            )
            
            X_all[i] = result['historical']
            Y_all[i] = result['future']
            
            metadata_all.append({
                'static_vacc_rate': static_vacc_rate,
                'static_vacc_delay': static_vacc_delay,
                'static_isol_rate': static_isol_rate,
                'static_isol_delay': static_isol_delay,
                'dynamic_vacc_rates': dynamic_vacc_rates,
                'dynamic_isol_rates': dynamic_isol_rates,
                'init_infected': init_infected,
            })
        
        return X_all, Y_all, metadata_all
