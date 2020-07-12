"""Storage for nested sampling points.

The information stored is a table with

- the likelihood threshold drawn from
- the likelihood, prior volume coordinates and physical coordinates of the point

"""

from __future__ import print_function, division
import numpy as np
import warnings
import os


class NullPointStore(object):
    """No storage."""

    def __init__(self, ncols):
        """Mock initialisation."""
        self.ncols = int(ncols)
        self.nrows = 0
        self.stack_empty = True
        self.ncalls = 0

    def reset(self):
        """Do nothing."""
        pass

    def close(self):
        """Do nothing."""
        pass

    def flush(self):
        """Do nothing."""
        pass

    def add(self, row, ncalls):
        """Increases the number of "stored" points."""
        self.nrows += 1
        self.ncalls = ncalls
        return self.nrows - 1

    def pop(self, Lmin):
        """Return no point (None, None)."""
        return None, None


class FilePointStore(object):
    """Base class for storing points in a file."""

    def reset(self):
        """Reset stack to loaded data.

        Useful when Lmin is not reset to a lower value.
        """
        # self.stack = sorted(self.stack + self.data, key=lambda e: (e[1][0], e[0]))
        self.stack_empty = len(self.stack) == 0
        # print("PointStore: have %d items" % len(self.stack))

    def close(self):
        """Close file."""
        self.fileobj.close()

    def flush(self):
        """Flush file to disk."""
        self.fileobj.flush()

    def pop(self, Lmin):
        """Request from the storage a point sampled from <= Lmin with L > Lmin.

        Returns
        -------
        index: int
            index of the point, None if no point exists
        point: array
            point values, None if no point exists

        """
        if self.stack_empty:
            return None, None

        # look forward to see if there is an exact match
        # if we do not use the exact matches
        #   this causes a shift in the loglikelihoods
        for i, (idx, next_row) in enumerate(self.stack):
            row_Lmin = next_row[0]
            L = next_row[1]
            if row_Lmin <= Lmin and L > Lmin:
                idx, row = self.stack.pop(i)
                self.stack_empty = self.stack == []
                return idx, row

        self.stack_empty = len(self.stack) == 0
        return None, None


class TextPointStore(FilePointStore):
    """Storage in a text file.

    Stores previously drawn points above some likelihood contour,
    so that they can be reused in another run.

    The format is a tab separated text file.
    Through the fmt and delimiter attributes the output can be altered.
    """

    def __init__(self, filepath, ncols):
        """Load and append to storage at *filepath*.

        The file should contain *ncols* columns (Lmin, L, and others).
        """
        self.ncols = int(ncols)
        self.nrows = 0
        self.stack_empty = True
        self._load(filepath)
        self.fileobj = open(filepath, 'ab')
        self.fmt = '%.18e'
        self.delimiter = '\t'

    def _load(self, filepath):
        """Load from data file *filepath*."""
        stack = []
        if os.path.exists(filepath):
            try:
                for line in open(filepath):
                    try:
                        parts = [float(p) for p in line.split()]
                        if len(parts) != self.ncols:
                            warnings.warn("skipping lines in '%s' with different number of columns" % (filepath))
                            continue
                        stack.append(parts)
                    except ValueError:
                        warnings.warn("skipping unparsable line in '%s'" % (filepath))
            except IOError:
                pass

        self.stack = list(enumerate(stack))
        self.ncalls = len(self.stack)
        self.reset()

    def add(self, row, ncalls):
        r"""Add data point *row* = [Lmin, L, \*otherinfo] to storage."""
        if len(row) != self.ncols:
            raise ValueError("expected %d values, got %d in %s" % (self.ncols, len(row), row))
        np.savetxt(self.fileobj, [row], fmt=self.fmt, delimiter=self.delimiter)
        self.nrows += 1
        self.ncalls = ncalls
        return self.nrows - 1


class HDF5PointStore(FilePointStore):
    """Storage in a HDF5 file.

    Stores previously drawn points above some likelihood contour,
    so that they can be reused in another run.

    The format is a HDF5 file, which grows as needed.
    """

    def __init__(self, filepath, ncols, group='/', **h5_file_args):
        """Load and append to storage at filepath.

        File contains *ncols* columns in 'points' dataset (Lmin, L, and others).
        h5_file_args are passed on to hdf5.File.
        """
        import h5py
        self.ncols = int(ncols)
        self.stack_empty = True
        h5_file_args['mode'] = h5_file_args.get('mode', 'a')
        self.fileobj = h5py.File(filepath, **h5_file_args)
        self.point_path = os.path.join(group, 'points')
        self._load()

    def _load(self):
        """Load from data file."""
        if self.point_path not in self.fileobj:
            self.fileobj.create_dataset(
                self.point_path, dtype=np.float,
                shape=(0, self.ncols), maxshape=(None, self.ncols))

        self.nrows, ncols = self.fileobj[self.point_path].shape
        if ncols != self.ncols:
            raise IOError("Tried to resume from file '%s', which has a different number of columns!" % (self.fileobj))
        points = self.fileobj[self.point_path][:]
        self.stack = list(enumerate(points))
        self.ncalls = self.fileobj.attrs.get('ncalls', len(self.stack))
        self.reset()

    def add(self, row, ncalls):
        """Add data point row = [Lmin, L, *otherinfo* to storage."""
        if len(row) != self.ncols:
            raise ValueError("expected %d values, got %d in %s" % (self.ncols, len(row), row))

        # make space:
        self.fileobj[self.point_path].resize(self.nrows + 1, axis=0)
        # insert:
        self.fileobj[self.point_path][self.nrows,:] = row
        if self.ncalls != ncalls:
            self.ncalls = self.fileobj.attrs['ncalls'] = ncalls

        self.nrows += 1
        return self.nrows - 1
