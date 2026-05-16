"""Timing and benchmarking infrastructure.

Collects per-operation timing data with machine metadata for reproducibility.
All experiments should use this to record training costs.
"""

import time
import platform
import os
import json


def machine_metadata():
    """Collect machine metadata for timing context."""
    meta = {
        'hostname': platform.node(),
        'platform': platform.platform(),
        'python': platform.python_version(),
    }
    # GPU info
    try:
        import subprocess
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.total',
             '--format=csv,noheader'], text=True, timeout=5)
        meta['gpus'] = [line.strip() for line in out.strip().split('\n')]
        vis = os.environ.get('CUDA_VISIBLE_DEVICES', 'all')
        meta['cuda_visible_devices'] = vis
    except Exception:
        meta['gpus'] = []
    # JAX device
    try:
        import jax
        meta['jax_devices'] = [str(d) for d in jax.devices()]
    except Exception:
        pass
    return meta


class Timer:
    """Context manager for timing code blocks.

    Usage:
        timer = Timer()
        with timer.section('e_step'):
            ...
        with timer.section('m_step'):
            ...
        timer.record('n_pairs', 50)
        timer.save('timing.json')
    """

    def __init__(self, metadata=None):
        self.records = []
        self.scalars = {}
        self.metadata = metadata or machine_metadata()
        self._section_name = None
        self._section_start = None

    class _Section:
        def __init__(self, timer, name):
            self.timer = timer
            self.name = name
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *args):
            elapsed = time.perf_counter() - self.start
            self.timer.records.append({
                'section': self.name,
                'elapsed_s': elapsed,
                'timestamp': time.time(),
            })
            self.elapsed = elapsed

    def section(self, name):
        return self._Section(self, name)

    def record(self, key, value):
        """Record a scalar metric."""
        self.scalars[key] = value

    def get_sections(self, name):
        """Get all timing records for a section name."""
        return [r for r in self.records if r['section'] == name]

    def total_time(self, name=None):
        """Total elapsed time for a section (or all sections)."""
        recs = self.get_sections(name) if name else self.records
        return sum(r['elapsed_s'] for r in recs)

    def summary(self):
        """Return a summary dict suitable for JSON serialization."""
        sections = {}
        for r in self.records:
            name = r['section']
            if name not in sections:
                sections[name] = {'count': 0, 'total_s': 0.0}
            sections[name]['count'] += 1
            sections[name]['total_s'] += r['elapsed_s']
        for v in sections.values():
            v['mean_s'] = v['total_s'] / v['count'] if v['count'] > 0 else 0
        return {
            'metadata': self.metadata,
            'scalars': self.scalars,
            'sections': sections,
            'total_s': sum(r['elapsed_s'] for r in self.records),
        }

    def save(self, path):
        """Save timing data to JSON."""
        with open(path, 'w') as f:
            json.dump(self.summary(), f, indent=2)

    def print_summary(self, prefix=''):
        """Print a brief summary to stdout."""
        s = self.summary()
        print(f"{prefix}Total time: {s['total_s']:.1f}s")
        for name, sec in s['sections'].items():
            print(f"{prefix}  {name}: {sec['count']}x, "
                  f"total={sec['total_s']:.1f}s, mean={sec['mean_s']:.3f}s")
        for k, v in s['scalars'].items():
            print(f"{prefix}  {k}: {v}")
