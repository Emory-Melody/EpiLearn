Visualization
===================================

``epilearn.visualize`` provides two low-level plotting helpers, unchanged in 0.1.0:
:func:`~epilearn.visualize.plot.plot_series` for time series and
:func:`~epilearn.visualize.plot.plot_graph` for a node-coloured network. Both are
re-exported at package level, so ``from epilearn.visualize import plot_series,
plot_graph`` works.

.. note::
   Neither helper calls ``plt.show()``: the call inside ``plot_graph`` is commented
   out so the module stays usable in headless environments. Nothing appears on
   screen — save the figure yourself (or rely on the inline renderer in a
   notebook).

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from epilearn.data import Dataset
    from epilearn.visualize import plot_series, plot_graph

    dataset = Dataset()
    dataset.load_toy_dataset()

    # --- time series: one line per column ---------------------------------
    series = dataset.y[:, :3].numpy()          # first three regions
    plot_series(series, columns=['region 0', 'region 1', 'region 2'],
                fig_size=(12, 4))
    # -> writes ./plot_series.png in the current working directory

    # --- network: plot_graph wants an EDGE INDEX, not an adjacency matrix --
    edge_index = np.array(np.nonzero(dataset.graph.numpy()))   # (2, n_edges)
    last_day = dataset.y[-1].numpy()
    states = (last_day > np.median(last_day)).astype(int)      # class per node

    pos = plot_graph(states, edge_index,
                     classes=['below median', 'above median'])
    plt.savefig('graph_last_day.png')          # plot_graph does not save for you
    print(len(pos))                            # 47 node positions, reusable layout

On the bundled toy dataset this prints ``47`` and produces ``plot_series.png``
(45 KB) and ``graph_last_day.png`` (880 KB).

Plot_Series
-----------

.. autofunction:: epilearn.visualize.plot.plot_series

.. warning::
   ``plot_series`` hard-codes its output path: the last line of the function is
   ``plt.savefig("plot_series.png")``. There is no ``save_path`` argument, so the
   figure always lands in ``./plot_series.png`` relative to the **current working
   directory**, and each call overwrites the previous one. To control where it
   goes, ``os.chdir()`` into the target directory first, or move the file
   afterwards.

   It also plots with ``seaborn.relplot``, which creates its own figure — so a
   preceding ``plt.figure(figsize=...)`` has no effect on the saved output and the
   ``fig_size`` argument is effectively ignored.

Plot_Graph
----------

.. autofunction:: epilearn.visualize.plot.plot_graph

.. warning::
   Three sharp edges in ``plot_graph``:

   * ``graph`` is an **edge index** of shape ``(2, n_edges)`` — column ``i`` is the
     edge ``(graph[0, i], graph[1, i])``. Passing an ``(N, N)`` adjacency matrix
     silently builds the wrong graph. Convert with
     ``np.array(np.nonzero(adjacency))``.
   * ``classes`` is declared optional but is required in practice: the body does
     ``labels[node] = classes[node]``, so leaving it as ``None`` raises
     ``TypeError: 'NoneType' object is not subscriptable``.
   * The internal colour map has only five entries (``0``–``4``), so every value in
     ``states`` must be an integer in ``0..4``; anything else raises ``KeyError``.

   The return value is the NetworkX position dict, which you can feed back in as
   ``layout=`` to keep the same node placement across frames.

Task-level plotting
-------------------

.. note::
   **Renamed in 0.1.0:** ``Forecast.plot_forecasts`` is now
   ``Forecast.plot_preds``. The old name no longer exists.

``Forecast.plot_preds`` draws predictions against targets together with the
conformal/adaptive uncertainty band:

.. code-block:: text

    task.plot_preds(eval_results, n_show=None, figsize=(15, 7), save_path=None,
                    backend='matplotlib', region_idx=0, horizon_idx=-1,
                    interactive=False)

``eval_results`` is the dict returned by ``task.evaluate_model(...)``; it must
contain ``predictions`` and ``targets`` (shape ``(time, regions, horizon)``), and
picks up ``adaptive_lower`` / ``adaptive_upper`` when present. Unlike the two
helpers above this one does take a ``save_path``, and it honours a ``'plotly'``
``backend`` for interactive HTML output. It returns ``(fig, ax)`` for matplotlib
and ``fig`` for plotly.
