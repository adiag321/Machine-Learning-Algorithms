import time
import pipeline
import monitor
import config

def run_simulation():
    print("==================================================")
    print("STARTING AUTOMATED FEEDBACK LOOP SIMULATION")
    print("==================================================")
    
    # Phase 1: Initialize System
    print("\n[Phase 1] Initializing system with clean baseline data...")
    base_data = pipeline.generate_base_data(n_samples=1200, shift_mean=0.0)
    pipeline.run_training_pipeline(base_data)
    
    time.sleep(1)
    
    # Phase 2: Steady State Production
    print("\n[Phase 2] Simulating steady-state production traffic...")
    prod_data_normal = pipeline.generate_base_data(n_samples=200, shift_mean=0.0)
    
    # Monitor evaluates production data
    needs_retraining = monitor.evaluate_production_performance(prod_data_normal)
    if not needs_retraining:
        print("System Healthy. Production operating within normal limits.")
    else:
        print("Retraining triggered prematurely.")
        
    time.sleep(1)
    
    # Phase 3: Concept/Data Drift Occurs
    print("\n[Phase 3] External shift occurs: Injecting feature drift into data stream...")
    prod_data_drifted = pipeline.generate_base_data(n_samples=300, shift_mean=1.5)
    
    # Monitor catches performance drop and statistical variance
    needs_retraining = monitor.evaluate_production_performance(prod_data_drifted)
    
    # Phase 4: Closing the Loop (Automated Action)
    if needs_retraining:
        print("\n[Phase 4] Automated trigger caught by orchestrator. Initiating loop closure...")
        # Ingesting fresh data (combining latest drifted window to adapt)
        new_training_batch = pipeline.generate_base_data(n_samples=1200, shift_mean=1.5)
        pipeline.run_training_pipeline(new_training_batch)
        
        print("\nRe-evaluating the new Champion model against drifted stream...")
        monitor.evaluate_production_performance(prod_data_drifted)
        print("Feedback loop successfully completed. System adapted to drift.")

if __name__ == "__main__":
    run_simulation()