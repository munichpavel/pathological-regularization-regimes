if __name__ == '__main__':
    """
    Usage example:

    python benchmark-multirun.py \
        --rel_data_parent outputs/data-2024-04-23/15-53-11 \
        --profile  # New optional flag
    """
    from pathlib import Path
    import os
    import subprocess
    import argparse
    from datetime import datetime
    import cProfile
    import pstats
    from zoneinfo import ZoneInfo

    # Initialize the ArgumentParser
    parser = argparse.ArgumentParser(description='Multirun script for model-fitting')

    # Add arguments
    parser.add_argument('--rel_data_parent', type=str, help='Data parent folder relative to repo root')
    parser.add_argument('--profile', action='store_true', help='Enable profiling')

    # Parse the arguments
    args = parser.parse_args()
    print(args)

    benchmark_report_out = Path(os.environ['REPO_ROOT']) / 'benchmarking-reports'
    benchmark_report_out.mkdir(exist_ok=True)

    berlin_tz = ZoneInfo('Europe/Berlin')
    start_time = datetime.now(berlin_tz)

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

    # Run the subprocess and measure execution time
    subprocess.run([
        'poetry', 'run', 'python', 'multirun-script.py',
        '--rel_data_parent', args.rel_data_parent
    ], text=True)

    end_time = datetime.now(berlin_tz)
    execution_time = (end_time - start_time).total_seconds()

    # Log the timestamp and execution time
    log_entry = f"Start Time: {start_time}, End Time: {end_time}, Execution Time: {execution_time} seconds\n"
    log_file = benchmark_report_out / 'execution_times.log'
    with open(log_file, 'a') as f:
        f.write(log_entry)

    if args.profile:
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumtime')
        stats_file = (benchmark_report_out / 'profile_stats.prof').as_posix()
        stats.dump_stats(stats_file)

        # Generate a readable report
        with open(benchmark_report_out / 'profile_report.txt', 'w') as stream:
            stats = pstats.Stats(stats_file, stream=stream)
            stats.sort_stats('cumtime')
            stats.print_stats()

    print(f"Execution completed. {log_entry}")
