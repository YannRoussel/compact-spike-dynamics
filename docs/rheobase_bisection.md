# Rheobase bisection

For a fixed current-step duration, rheobase is the lowest injected current that
elicits at least one action potential. In a finite deterministic simulation it
is a protocol-dependent firing boundary, not the idealized infinite-duration
rheobase.

## Algorithm

1. Start at 0 pA, which is the non-spiking lower bound for a stable resting
   model.
2. Test the lowest positive grid current first. This preserves very excitable
   models that spike at low current but enter depolarization block at the
   largest current.
3. If the lowest current does not spike, test the largest grid current and
   binary-search the ordered coarse grid for the first spiking grid point.
4. The adjacent non-spiking and spiking grid currents form
   `[I_low, I_high]`.
5. Test the continuous midpoint `I_mid = (I_low + I_high) / 2`.
6. If the midpoint spikes, replace `I_high` with `I_mid`. Otherwise replace
   `I_low` with `I_mid`.
7. Repeat until `I_high - I_low` is below the requested pA tolerance.

The reported rheobase is `I_high`, because it is the endpoint known to spike.
The lower endpoint, upper endpoint, bracket width, and number of iterations are
all stored as model features.

For canonical HH at 22 degrees C with a 600 ms step, the original grid reported
800 pA. Bisection refines this to a final bracket of approximately
765.6-767.2 pA.

## Why it affects latency

A coarse grid can record a waveform far above the true firing boundary. The
stronger current depolarizes the membrane faster and shortens first-spike
latency. Measuring at the bisected upper endpoint removes most of this current
quantization bias.

Bisection does not guarantee long latency. Several two-current HH models fire
almost immediately whenever they fire at all. In the current pilot, long
sweeps and bisection extend the modeled latency range but do not reproduce the
hundreds-of-milliseconds tail in the biological recordings.

## Assumptions

The binary grid search assumes monotonic first-spike onset after the lowest
current guard. The method also assumes local monotonicity around the first
firing boundary:
non-spiking below the boundary and spiking just above it. This need not hold at
very high currents because depolarization block can suppress spikes. Searching
the lowest current first preserves the depolarization-block cases observed in
the sampled HH population.

The binary and original sequential searches were compared on identical
128-model room-temperature and 256-model warm parameter sets. Accepted IDs,
rejected IDs, rheobases, and all finite features matched exactly. With six
workers, wall time fell by approximately five- to sevenfold.

The adaptive search stops when voltage crosses the spike criterion. Final
features are always extracted from a full fixed-step trace, and models whose
final waveform does not satisfy the full spike detector are rejected.
