"""
Progress tracking and status updates
"""

import sys
from datetime import datetime
from typing import Optional


class ProgressTracker:
    """Tracks and displays generation progress"""

    def __init__(self, total_entries: int, batch_size: int):
        self.total_entries = total_entries
        self.batch_size = batch_size
        self.current_entries = 0
        self.current_status = "Initializing"
        self.start_time = datetime.now()
        self.only_status = False

    def update_status(self, status: str):
        """Update current operation status"""
        self.current_status = status
        self._display_progress()

    def add_entries(self, count: int):
        """Add completed entries to count"""
        self.current_entries += count
        self._display_progress()

    def _display_progress(self):
        """Display current progress"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        elapsed_str = self._format_time(elapsed)

        if self.only_status:
            # Simple status-only display
            status_line = f"\rElapsed: {elapsed_str} | Status: {self.current_status}"
            # Print without newline
            sys.stdout.write("\r" + " " * 150)  # Clear line
            sys.stdout.write(status_line)
            sys.stdout.flush()
            return

        # Calculate progress percentage
        progress_pct = (
            (self.current_entries / self.total_entries * 100)
            if self.total_entries > 0
            else 0
        )

        # Create progress bar
        bar_length = 40
        filled_length = int(bar_length * progress_pct / 100)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)

        # Calculate ETA
        if self.current_entries > 0 and elapsed > 0:
            rate = self.current_entries / elapsed
            remaining = self.total_entries - self.current_entries
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "calculating..."

        # Format elapsed time
        elapsed_str = self._format_time(elapsed)

        # Build status line
        status_line = (
            f"\r[{bar}] {progress_pct:.1f}% | "
            f"{self.current_entries}/{self.total_entries} entries | "
            f"Elapsed: {elapsed_str} | ETA: {eta_str} | "
            f"Status: {self.current_status}"
        )

        # Print without newline
        sys.stdout.write("\r" + " " * 150)  # Clear line
        sys.stdout.write(status_line)
        sys.stdout.flush()

    def _format_time(self, seconds: float) -> str:
        """Format seconds into readable time string"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            mins = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds / 3600)
            mins = int((seconds % 3600) / 60)
            return f"{hours}h {mins}m"

    def complete(self):
        """Mark progress as complete"""
        self.current_entries = self.total_entries
        self.current_status = "Complete"
        self._display_progress()
        print()  # New line after completion


class DetailedProgressTracker(ProgressTracker):
    """Extended tracker with more detailed statistics"""

    def __init__(self, total_entries: int, batch_size: int):
        super().__init__(total_entries, batch_size)
        self.batches_completed = 0
        self.total_batches = (total_entries + batch_size - 1) // batch_size
        self.corrections_attempted = 0
        self.corrections_successful = 0
        self.entries_rejected = 0

    def increment_batch(self):
        """Increment completed batch count"""
        self.batches_completed += 1

    def add_correction_attempt(self, successful: bool):
        """Record a correction attempt"""
        self.corrections_attempted += 1
        if successful:
            self.corrections_successful += 1
        else:
            self.entries_rejected += 1

    def get_statistics(self) -> dict:
        """Get detailed statistics"""
        elapsed = (datetime.now() - self.start_time).total_seconds()

        return {
            "total_entries": self.total_entries,
            "completed_entries": self.current_entries,
            "progress_percentage": (
                (self.current_entries / self.total_entries * 100)
                if self.total_entries > 0
                else 0
            ),
            "batches_completed": self.batches_completed,
            "total_batches": self.total_batches,
            "corrections_attempted": self.corrections_attempted,
            "corrections_successful": self.corrections_successful,
            "entries_rejected": self.entries_rejected,
            "success_rate": (
                (self.corrections_successful / self.corrections_attempted * 100)
                if self.corrections_attempted > 0
                else 0
            ),
            "elapsed_time_seconds": elapsed,
            "entries_per_second": self.current_entries / elapsed if elapsed > 0 else 0,
        }

    def print_summary(self):
        """Print final summary"""
        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("GENERATION SUMMARY")
        print("=" * 60)
        print(
            f"Total Entries Generated: {stats['completed_entries']}/{stats['total_entries']}"
        )
        print(
            f"Batches Processed: {stats['batches_completed']}/{stats['total_batches']}"
        )
        print(f"Corrections Attempted: {stats['corrections_attempted']}")
        print(f"Corrections Successful: {stats['corrections_successful']}")
        print(f"Entries Rejected: {stats['entries_rejected']}")

        if stats["corrections_attempted"] > 0:
            print(f"Correction Success Rate: {stats['success_rate']:.1f}%")

        print(f"Total Time: {self._format_time(stats['elapsed_time_seconds'])}")
        print(f"Average Rate: {stats['entries_per_second']:.2f} entries/second")
        print("=" * 60)
