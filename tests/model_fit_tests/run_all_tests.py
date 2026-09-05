#!/usr/bin/env python
"""
Run all model fitting tests.

This script runs all model fit tests and generates a comprehensive report.
Each test verifies that models can actually learn patterns from data,
not just that their forward passes work.

Usage:
    python run_all_tests.py           # Run all tests
    python run_all_tests.py temporal  # Run only temporal tests
    python run_all_tests.py sklearn   # Run only sklearn tests
    python run_all_tests.py stats     # Run only statsmodels tests
    python run_all_tests.py spatial   # Run only spatiotemporal tests
"""
import sys
import os
import time
import json

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from test_config import OUTPUT_DIR


def run_temporal_nn_tests():
    """Run temporal neural network tests."""
    print("\n" + "=" * 80)
    print("RUNNING: Temporal Neural Network Model Tests")
    print("=" * 80)
    from test_temporal_nn import test_temporal_nn_models
    return test_temporal_nn_models()


def run_sklearn_tests():
    """Run sklearn model tests."""
    print("\n" + "=" * 80)
    print("RUNNING: Scikit-Learn Model Tests")
    print("=" * 80)
    from test_sklearn_models import test_sklearn_models
    return test_sklearn_models()


def run_statsmodels_tests():
    """Run statsmodels tests."""
    print("\n" + "=" * 80)
    print("RUNNING: StatsModels Tests")
    print("=" * 80)
    from test_statsmodels import test_statsmodels
    return test_statsmodels()


def run_spatiotemporal_tests():
    """Run spatiotemporal model tests."""
    print("\n" + "=" * 80)
    print("RUNNING: Spatiotemporal Model Tests")
    print("=" * 80)
    from test_spatiotemporal import test_spatiotemporal_models
    return test_spatiotemporal_models()


def print_final_report(all_results):
    """Print a comprehensive final report."""
    print("\n" + "=" * 80)
    print("FINAL REPORT: Model Fitting Tests")
    print("=" * 80)
    
    total_pass = 0
    total_warn = 0
    total_fail = 0
    
    for category, results in all_results.items():
        print(f"\n{category}:")
        print("-" * 40)
        
        if results is None:
            print("  (skipped)")
            continue
            
        for model_name, result in results.items():
            status = result.get('status', 'UNKNOWN')
            
            if status == 'PASS':
                total_pass += 1
                r2 = result.get('r2', 0)
                mse = result.get('mse', 0)
                print(f"  ✅ {model_name}: R²={r2:.3f}, MSE={mse:.4f}")
            elif status == 'WARN':
                total_warn += 1
                r2 = result.get('r2', 0)
                mse = result.get('mse', 0)
                print(f"  ⚠️  {model_name}: R²={r2:.3f}, MSE={mse:.4f}")
            else:
                total_fail += 1
                error = result.get('error', 'Unknown error')
                print(f"  ❌ {model_name}: {error[:50]}...")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  ✅ Passed: {total_pass}")
    print(f"  ⚠️  Warnings: {total_warn}")
    print(f"  ❌ Failed: {total_fail}")
    print(f"  📊 Total: {total_pass + total_warn + total_fail}")
    
    if total_fail == 0:
        print("\n🎉 All models can fit training data!")
    elif total_fail > 0:
        print(f"\n⚠️  {total_fail} models failed - investigate the errors above")
    
    # Save report to JSON
    report_path = os.path.join(OUTPUT_DIR, 'test_report.json')
    report = {
        'summary': {
            'passed': total_pass,
            'warnings': total_warn,
            'failed': total_fail,
            'total': total_pass + total_warn + total_fail
        },
        'results': {}
    }
    
    for category, results in all_results.items():
        if results:
            report['results'][category] = {
                model: {k: (v if not isinstance(v, float) or not (v != v) else None) 
                        for k, v in result.items()}
                for model, result in results.items()
            }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Report saved to: {report_path}")
    print(f"📁 Plots saved to: {OUTPUT_DIR}/")
    
    return total_fail == 0


def main():
    """Main entry point."""
    print("=" * 80)
    print("MODEL FITTING TESTS")
    print("Testing that models can actually learn patterns from data")
    print("=" * 80)
    
    start_time = time.time()
    
    # Check command line args for specific test category
    test_category = sys.argv[1].lower() if len(sys.argv) > 1 else 'all'
    
    all_results = {}
    
    if test_category in ['all', 'temporal']:
        all_results['Temporal Neural Networks'] = run_temporal_nn_tests()
    
    if test_category in ['all', 'sklearn']:
        all_results['Scikit-Learn Models'] = run_sklearn_tests()
    
    if test_category in ['all', 'stats']:
        all_results['StatsModels'] = run_statsmodels_tests()
    
    if test_category in ['all', 'spatial']:
        all_results['Spatiotemporal Models'] = run_spatiotemporal_tests()
    
    elapsed = time.time() - start_time
    
    # Print final report
    success = print_final_report(all_results)
    
    print(f"\n⏱️  Total time: {elapsed:.1f} seconds")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
