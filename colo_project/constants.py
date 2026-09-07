"""Path-loss model constants shared by the forward model and the CRLB.

These live here rather than in graph_utils or metrics so that neither module has
to import the other. Changing them changes the measurement model; the values
actually used to generate a scenario are stored in its .npz.
"""

# Path-loss exponent.
ALPHA = 3.15

# Reference distance for the log-distance path-loss model, in meters.
D0 = 1.15


# Independent RNG streams derived from a scenario seed, so that changing one
# stage (e.g. n_iter) cannot perturb the draws of another.
STREAM_DATA = 0
STREAM_NBP = 1
