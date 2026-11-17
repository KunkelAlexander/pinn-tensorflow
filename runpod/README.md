# Runpod GPU setup (RTX 4090 + GAMER)

GAMER (gamer-project/gamer, GPU-accelerated AMR) is the reference code that the `toy_models/` findings compare
against (see `toy_models/RESEARCH_DIRECTIONS.md`, "GAMER's gravity path"). This folder records how it was set up
on a Runpod pod so it can be rebuilt and re-run. `setup_gamer.sh` builds it (see "Build").

## 1. Connecting Claude Code to Runpod

Done once, following https://agents.runpod.io/:

```
claude plugin marketplace add runpod/runpod-plugins-official
claude plugin install runpod@runpod
```

Then `/reload-plugins`, **restart Claude Code** (plugin MCP changes only take effect in a new session), and
`/mcp` -> runpod -> *Sign in with Runpod* (OAuth, no key on disk). Sign in with the account that owns the pod, or
`list-pods` returns an empty list. If OAuth is rejected, use an API key from https://console.runpod.io/user/settings
as `RUNPOD_API_KEY` (never write it to a file in this repo).

What the MCP tools can and cannot do here:
- They manage infrastructure (pods, volumes, billing) and `exec-pod` runs ONE command in a running pod.
  Each call waits at most 25 s, so start long jobs with `nohup ... > log 2>&1 &` and poll the log.
- No SSH session, no file transfer to or from the pod, no interactive terminal. Output files (HDF5 snapshots,
  PNGs) stay on the pod. To get them out, use the Jupyter file browser/terminal or `runpodctl`.

## 2. The pod (created 2026-11-15)

| | |
|---|---|
| Name / id | `pinn-4090` / `3tsp51kn7xfrce` |
| GPU | 1x RTX 4090 (24 GB), secure cloud, data center EUR-IS-2; 64 visible cores, ~500 GB RAM |
| Price | $0.89/hr while running |
| Template / image | `runpod-torch-v280` / `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (CUDA 12.8 toolkit incl. `nvcc`, driver CUDA 13.2, gcc 13, Python 3.12) |
| Disks | 30 GB container disk (wiped on restart), 50 GB persistent `/workspace` (network filesystem) |
| Access | Jupyter on port 8888 (`https://<podId>-8888.proxy.runpod.net`; the password is the `JUPYTER_PASSWORD` env var, see console -> Connect). No SSH key registered. |

Billing: stopping ends the GPU charge but the `/workspace` disk keeps billing; terminating deletes `/workspace`.
4090 stock was LOW, so a stopped pod may not get its GPU back on restart. Always check `df -h /workspace`:
the LSS and halo runs write several GB per snapshot.

## 3. Required packages

Installed with apt (lost on every pod restart, so re-run `setup_gamer.sh` after one):

| Package | Why |
|---|---|
| `libfftw3-dev libfftw3-mpi-dev` | FFTW3 for gravity/FFT solvers. The **`-mpi-dev`** package is required for MPI builds (`fftw3-mpi.h`). |
| `libhdf5-dev` | HDF5 snapshots (`Data_NNNNNN`) |
| `libgsl-dev` | GSL, required by the hybrid build (`--gsl=true`) |
| `openmpi-bin libopenmpi-dev` | MPI (`mpicxx`, `mpirun`) |

Already in the image: `nvcc` 12.8, `gcc/g++`, `make`, `git`, `python3`, `curl`.
Python (pip, only for the analysis scripts): `yt` 4.4.4 and `h5py`. yt prints "Could not enable parallelism:
mpi4py is not installed" and then runs serially; that is fine.

## 4. Build

Run inside the pod: `bash /workspace/<repo>/runpod/setup_gamer.sh wave` or `... hybrid`. (This script was edited
after the builds below and has not been re-run end to end; the commands in it are the ones that were run.)

What it does, and why:
- Clones the **fork** `https://github.com/KunkelAlexander/gamer.git`, branch `fd4-poisson`, to `/workspace/gamer`
  (see "GAMER source and the 4th-order Poisson solver" below). Override with `GAMER_REPO` / `GAMER_BRANCH`; plain
  upstream is `GAMER_REPO=https://github.com/gamer-project/gamer.git GAMER_BRANCH=main`. An existing clone is switched to
  the branch and fast-forwarded. Base commit of the branch: upstream `e36797538` (2026-10-12).
- Debian installs HDF5 and OpenMPI in non-standard directories, so `/workspace/deps/{hdf5,mpi}` hold symlinks to
  `include/`, `lib/`, `bin/` that GAMER's `*_PATH` settings can point at.
- Writes the machine config `/workspace/gamer/configs/runpod_4090.config`: g++/mpicxx, CUDA at `/usr/local/cuda`,
  `GPU_COMPUTE_CAPABILITY 890`, and **`-std=gnu++14`**. Do not use strict `-std=c++14`: the GramFE tables use GNU
  `q` (`__float128`) literals and the hybrid build fails with `unable to find numeric literal operator 'operator""q'`.
- Configures from `src/` with `python3 configure.py --machine=runpod_4090 ...`, then `make -j`. The options are the
  ones in each test problem's `generate_make.sh`:

| Flavor | configure options (after `--mpi=true --hdf5=true --fftw=FFTW3 --gpu=true --model=ELBDM --gravity=true`) | Test problems |
|---|---|---|
| `wave` | none | Soliton, IsolatedHalo, (HaloMerger, ...) |
| `gravity` | (separate options: `--gpu=true --hdf5=true --fftw=FFTW3 --model=HYDRO --gravity=true`, no MPI/ELBDM) | Hydro/Gravity (Poisson-solver error vs analytic potential) |
| `hybrid` | `--elbdm_scheme=HYBRID --wave_scheme=GRAMFE --gramfe_scheme=MATMUL --comoving=true --gsl=true --spectral_interpolation=true` | LSS_Hybrid, LSS_Hybrid_Zoomin |

- Building one flavor overwrites `/workspace/gamer/bin/gamer`, so the script copies it to
  `/workspace/gamer_<flavor>_bin`. Always run the flavor-specific copy, never `bin/gamer`. The wave build on the pod is
  `/workspace/gamer_isolated_bin`, the hybrid one `/workspace/gamer_hybrid_bin`. A full build takes about 3-4 minutes.

### GAMER source and the 4th-order Poisson solver

Code changes to GAMER live in a fork, not on the pod:

- Fork: https://github.com/KunkelAlexander/gamer (public). Local clone: `C:/Users/TE/Desktop/Privat/Projects/gamer`
  (outside this repo; remotes `origin` = fork, `upstream` = gamer-project/gamer). Branch `fd4-poisson`, based on `e36797538`.
- Workflow: edit and commit locally, `git push`, then on the pod `cd /workspace/gamer && git fetch origin && git merge --ff-only origin/fd4-poisson`
  and rebuild. The Runpod tools have no file transfer, so git is the only route. Never leave edits only on the pod.
- Plan and design: `runpod/FD4_POISSON_PLAN.md`. Implemented (3 commits): input parameter `POT_STENCIL_ORDER` (2 default | 4),
  FD4 symbol in the periodic FFT, 4th-order SOR by deferred correction (GPU kernel + CPU twin). With order 2 the results are
  unchanged; the parameter is not stored in the HDF5/restart tables (it is in `Record__Note`).
- Unit-test results (GPU==CPU, 9x error reduction on a periodic smooth test, no gain on isolated Plummer, point-value interpolation schemes worse): `runpod/FD4_POISSON_PLAN.md` section 6. LSS_Hybrid with order 4 not run yet.
- `POT_STENCIL_ORDER 4` limits: SOR Poisson solver only (not MG), not GPU+FLOAT8. The isolated-BC base level (continuum
  Green function) is untouched; only periodic base levels get the FD4 symbol.
- Low-priority build while a timed run is going: `/workspace/fd4_scripts/build_fd4.sh` (nice, make -j8) writes
  `/workspace/gamer_gravity_fd4_bin` and `/workspace/gamer_hybrid_fd4_bin`.

## 5. Running (all problems)

```
cp -r /workspace/gamer/example/test_problem/ELBDM/<Problem> /workspace/<run_dir>
cd /workspace/<run_dir> && cp /workspace/gamer_<flavor>_bin gamer
# edit Input__Parameter: set OMP_NTHREAD 8 (see below)
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1     # the pod user is root
nohup mpirun -np 1 --bind-to none ./gamer > log 2>&1 &
grep -c '^Time:' log      # one line per step
```

Pitfalls found:
- **`OMP_NTHREAD`.** At `-1` with Open MPI's default binding, GAMER resets it to 2 threads (about 40 s/step on the
  512^3 halo). With `--bind-to none` it picks 64 threads, which was slower still (the initial Poisson solve did not
  finish in 6+ minutes). **Set `OMP_NTHREAD 8`** in `Input__Parameter` and use `--bind-to none`.
- `END_T`/`END_STEP` can be -1 ("set by the test problem") and then the run never ends (Soliton kept going to t>700
  and wrote 36+ snapshots until killed). Check them before launching, and kill runs yourself (`pkill -x gamer`).
- ICs come from `hub.yt` via each problem's `download_*.sh`, which verify a SHA256 ("File broken" is printed on mismatch).

## 6. Test problems run so far

### Soliton
Compiled and ran; tiny (GPU ~5%, 704 MiB). Smoke test only.

### IsolatedHalo (wave build): uniform grid vs AMR (run 2026-11-15)
`example/test_problem/ELBDM/IsolatedHalo` has the halo density-profile script `yt_script/plot__profile.py`.
Setup: IC `UM_IC_run05-halo08-lv4` (800 MB, `sh download_ic.sh`), 512^3 base grid, box 0.175 Mpc/h, isolated gravity
BC, ELBDM mass 8e-23 eV. Set `END_T 5.7116620e-03`, `END_STEP 32`, `OUTPUT_DT 2.8558310e-03` (3 snapshots).
Run dirs: `/workspace/halo_run` (`MAX_LEVEL 0`, uniform; 7.2 GB) and `/workspace/halo_amr`
(`MAX_LEVEL 2`, `OPT__FLAG_RHO 1`, thresholds 1e3 and 4e3 in the shipped `Input__Flag_Rho`).
AMR at t=0: level 0 262144 patches, level 1 31728 (1.5% of the volume), level 2 35200 (0.2%).

Profile via `cd yt_script && python3 plot__profile.py -p .. -s 0 -e 2` (PNGs stay on the pod). Numeric AMR/uniform
ratio of the volume-weighted density at the final time (`/workspace/compare_amr.py 2`, 32 log bins 0.5-100 kpc):

| r (kpc) | 1.06 | 1.47 | 2.05 | 2.85 | 5.53 | 10.7 | 20.8 | 40.4 | 78.3 |
|---|---|---|---|---|---|---|---|---|---|
| AMR/uniform | 0.979 | 1.055 | 1.049 | 1.004 | 0.997 | 0.999 | 1.005 | 1.000 | 1.000 |

Within 1% outside 3 kpc and about 5% at 1.5-2 kpc. This is only 32 steps of a single halo: it shows the pipeline works
and that AMR does not visibly change the profile over that time. It is not a reproduction of the mismatch.

### LSS_Hybrid (hybrid build): light and heavy halo, the cosmological test
`example/test_problem/ELBDM/LSS_Hybrid` (README in that folder). ELBDM mass 2e-23 eV, box 2.8 Mpc/h, **base grid 64^3**,
`MAX_LEVEL 6`, fluid scheme on levels 0-3 and wave scheme from `ELBDM_FIRST_WAVE_LEVEL 4` (2.75 ckpc/h), refinement by
`OPT__FLAG_SPECTRAL` and `OPT__FLAG_INTERFERENCE` (`Input__Flag_Rho` is not used: `OPT__FLAG_RHO 0`). The default
maximum resolution is only ~0.7 ckpc/h. Output mode 3: snapshots at the times in `Input__DumpTable`
(the second column is the scale factor a; ID 45 is a = 0.5303239, z = 0.886).

Setup done on the pod (run dirs `/workspace/lss_light`, `/workspace/lss_heavy`):
1. `sh download_light_halo_ic.sh` / `sh download_heavy_halo_ic.sh`. Light downloads a 64^3 MUSIC IC and rescales it to
   256^3 (`elbdm_rescale_periodic_IC.py`), then both convert wave IC to hybrid IC (`elbdm_wave_to_hybrid_IC.py`) and link
   `UM_IC`. The IC is read at level 2 (256^3), downgraded to level 0 (`OPT__UM_IC_DOWNGRADE 1`) and refined back up.
2. `sh download_spectral_interpolation_tables.sh` (needed for `OPT__FLU_INT_SCHEME -1`/`8`); the heavy dir symlinks the
   light dir's `interpolation_tables` and `boundary2extension_tables`.
3. `OMP_NTHREAD 8`. `Input__DumpTable` cut after table ID 45 (a = 0.5303, z = 0.886) so the run stops there instead
   of going to z = 0 (original kept as `Input__DumpTable.orig`).
   **Dump numbering gotcha:** GAMER ignores the "Dump ID" column of the table (`sscanf` into a throwaway variable) and
   numbers files `Data_NNNNNN` by a running counter. `Data_000000` is the initial state (a = 0.0099), and table rows
   with a earlier than the start (IDs -6 to -3) are skipped. So **`Data_000045` is table ID 42 (a = 0.4857, z = 1.059)
   and table ID 45 is `Data_000048`**. Trimming the table by dropping rows changes which file is which, so cut
   only at the end. Which one "dump 45" means has to be confirmed.

Status: see section 8.

## 7. What is documented about the halo mismatch (checked 2026-11-15)

- **Paper:** arXiv:2412.09908, Liao et al., "Deciphering the Soliton-Halo Relation in Fuzzy Dark Matter" (read via its
  HTML version). It documents a discrepancy between a low-resolution FDM run and higher-resolution FDM/N-body runs in
  the halo density profile (Fig. 2, App. A.3 / Fig. S3): an overly concentrated halo and a soliton about 30% more
  massive, at r below ~7 kpc for a 9.8e11 Msun halo at m22 = 0.2. It names no radius range for the deviation and gives no
  AMR-vs-uniform or potential/force comparison. It gives no test problem, input files, base resolution, `MAX_LEVEL` or
  refinement thresholds, and the Supplemental Material URL is a placeholder. **Its explanation (underestimated quantum
  pressure) is the author's own and is NOT to be taken as established**: a lower-resolution run coarsens the wave and
  the gravity solver together, so the paper cannot separate the two.
- **GAMER wiki / `doc/wiki`:** no page on halo-radius gravity accuracy or AMR-vs-uniform agreement. Closest: the
  Perturbation test ("obviously different" from a 1024^2 reference; refinement settings "need modifications"),
  the hybrid "Spurious Halos" and "Sudden Overrefinement" notes, and the spectral-interpolation known issues.
- **`LSS_Hybrid` README:** states the 64^3 base resolution "has shown very good agreement with higher resolution wave
  initial conditions", that `ELBDM_FIRST_WAVE_LEVEL` too low "will negatively affect the accuracy", and that the default
  maximum resolution can only marginally resolve a central soliton. It does not describe a mismatch or ship a
  high-resolution reference. **A high-resolution reference run (higher base resolution or deeper `MAX_LEVEL`) has to
  be created by us.**
- **`RestrictionMismatch`:** a phase mismatch at coarse-fine boundaries from restriction/interpolation (`OPT__INT_PHASE`,
  `OPT__RES_PHASE`) in a Gaussian wave packet. Not a gravity test.
- `LSS_Hybrid_Zoomin` (Chan et al. 2025, MNRAS 540, 2653) also has light/heavy halo ICs (`download_IC_lighthalo.sh`,
  `download_IC_heavyhalo.sh`), 10 Mpc/h box, 74 GB of snapshots by default, `OPT__FLAG_REGION 1`.

## 7b. Long LSS_Hybrid runs: slowdown near a = 1, intermediate dumps, disk cleanup

(From the repo owner, from experience with these runs; not yet measured by us except where noted.)

- **The runs slow down badly towards a = 1**, when the halo forms and refinement explodes. The default light-halo
  settings (interference + spectral refinement only) are fine. A run with **density refinement on more levels
  (`OPT__FLAG_RHO 1`) can take hours**, not minutes.
- **Plan long runs with intermediate dumps, and clean them up while the run goes**, so that disk stays bounded and a
  crash or a too-slow run still leaves something to analyse:
  1. In `Input__DumpTable`, list a handful of times (e.g. a = 0.5, 0.7, 0.85, 0.95, 1.0) instead of only a = 1.
     Remember the numbering gotcha in section 6: files are numbered by a running counter (`Data_000000` is the initial
     state), not by the ID column. Write the counter-to-a mapping into the run's notes.
  2. Run a background cleanup loop next to the simulation that computes the profile of each new snapshot and then
     deletes everything except the most recent snapshot and the final one. Example (run it with `nohup ... &` from the
     run directory; it assumes `yt_script/plot__profile.py`, which writes `profile_<Data_N>.txt`):

     ```
     while pgrep -x gamer > /dev/null; do
       for d in $(ls -d Data_0* 2>/dev/null | sort | head -n -1); do      # all but the newest snapshot
         [ -f yt_script/profile_$d.txt ] || (cd yt_script && python3 plot__profile.py -p .. -s ${d#Data_} -e ${d#Data_})
         [ -f yt_script/profile_$d.txt ] && rm -rf "$d" Diag_${d#Data_}   # delete only after its profile exists
       done
       sleep 60
     done
     ```
     (Needs `OPT__OUTPUT_TOTAL 1`. Also delete the `PowerSpec_*` files when no longer needed. They are small.)
  3. Check disk (`df -h /workspace`, `du -sh /workspace`) before each launch. Dumps were 4 MB at a = 0.01 and 15 MB at
     a = 0.14 in the heavy-halo test, and grow with the number of patches, so the late ones are the big ones.
- Always **time** runs: `run_timed.sh` in `/workspace/lss_light` writes `START`, `END`, `ELAPSED_SECONDS` and the exit
  code to `timing.txt`. Wall-clock times are only comparable when nothing else uses the GPU/CPU at the same time.

## 8. LSS_Hybrid run status (end of session 1, 2026-11-15, ~14:40 UTC)

### Light halo, default refinement (interference + spectral, no density refinement), hybrid build, to a = 1

- **KILLED: the pod was stopped at the end of session 1 (about 14:40 UTC, step ~497, a = 0.745) at the user's request, so this run did not finish and left no
  snapshot, no profile and no timing. It has to be restarted (see section 9).** The description below is what it was at that point.
- Run dir `/workspace/lss_light`, started 13:43:32 UTC by `run_timed.sh` (hybrid binary `/workspace/gamer_hybrid_bin`, `OMP_NTHREAD 8`,
  only dump = a = 1). At 14:36 it was at step 496, a = 0.740, 53 min in, with 89k patches; `timing.txt` has only `START` until it ends.
- `post_light.sh` (queued, waits for `ELAPSED` in `timing.txt`) then writes `post.log` (snapshot size), runs `yt_script/plot__profile.py`, copies
  `/workspace/profile_light_default.txt` and `.png`, deletes `Data_000000` and, if > 1 GB, `Data_000001`. **If the pod was stopped before the run
  ended, the run is lost (no checkpoints exist) and has to be restarted; nothing of it is saved.** The profile script was only tested on `Data_000000`
  (no halo yet), so check the centre/radial range on the real snapshot.
- Elapsed time of this run is inflated: a heavy-halo run overlapped the first part, and builds and unit tests (nice'd) ran next to it.

### Why it is slow (measured, cause inferred)

- `Record__Timing` (step 494, 139 s): `Flu_Adv` 88 %, `Gra_Adv` (Poisson) 7 %; level 6 alone 90 s (22.8k patches, many sub-steps). Levels 4-6 are
  the wave (Gram-FE) levels.
- `Record__Performance`: ~8e6 cell-updates/s, constant from step 474 to 495; the step time grew 23 s -> 139 s only because the cell-updates per step grew
  1.5e8 -> 1.15e9 (more fine-level sub-steps).
- GPU utilisation 0-7 % while the 8 OpenMP threads are at 90-100 %: CPU-bound. A 4090 should be orders of magnitude faster per cell for the wave solver.
- Likely cause (not confirmed): `GRAMFE_FLU_NXT 64`, i.e. each 16^3 patch group is fed a 64^3 array (64x the payload, almost all ghost zones) that is prepared
  on the CPU inside `Flu_Adv` (copies and interpolation from coarser levels).
- Could not profile: `gdb` cannot attach in the container (ptrace blocked; gdb is installed anyway, lost on pod restart), and the build has `TIMING_SOLVER` off.
- Estimate at 8 threads: ~2.5 h from a = 0.74 to 1.
- **Correction found when stopping the pod:** `get-pod` reports `gpu.vcpuCount 12`, i.e. the pod is allotted **12 vCPUs** although `nproc` shows 64 (the host's cores;
  no cgroup limit file is visible inside). So 8 threads already use most of the allotment (cpu util 78 %), more than ~12 OpenMP threads cannot help and oversubscription
  explains why 64 threads thrashed earlier. The "try 24-32 threads" idea in the first version of the checklist is wrong; faster runs need either a pod/GPU type with more vCPUs
  (check `list-gpu-types`/`get-gpu-type` vcpu counts, or a CPU-heavier template) or less CPU work per step (e.g. a smaller `GRAMFE_FLU_NXT`/ghost-zone setup, not explored).

### Not started yet

- Density-refinement run (`OPT__FLAG_RHO 1`, shipped `Input__Flag_Rho`, 8x per level). Expect it to be slower than the run above. Use section 7b.
- LSS_Hybrid with `POT_STENCIL_ORDER 4` (binary `/workspace/gamer_hybrid_fd4_bin`, built from the fork branch). Only unit-tested so far.
- High-resolution reference and the wave-vs-gravity resolution scan.

## 9. Where we left off and what to do next

State of the pod `3tsp51kn7xfrce` ("pinn-4090", RTX 4090, EUR-IS-2, $0.89/h while running): **stopped** (not terminated) at the end of session 1 at the user's request,
which killed the light-halo run. A stopped pod keeps `/workspace` (volume disk, billed at the stopped-disk rate) but wipes the container disk (apt packages, gdb) and
may fail to start if another renter takes the GPU on that host (then: wait, start with 0 GPUs to reach the files, or create a new pod; `/workspace` is a volume of the
old pod, so check how to move it before terminating). Terminating deletes the volume.

Next session checklist:
1. `list-pods`; if the pod is stopped/gone, start it (or create one, section 2) and run `bash setup_gamer.sh hybrid` (re-installs apt packages, pulls the fork, rebuilds;
   `/workspace/deps`, binaries and run dirs survive a stop). Check `df -h /workspace`: it was 2.8 GB after tidying; heavy unit-test dirs reached 40 GB.
2. Check `/workspace/lss_light/{timing.txt,post.log}` and `/workspace/profile_light_default.{txt,png}`. Report the elapsed time (with the caveats above).
3. **Speed before the density-refinement run** (this decides the run time): the pod has only 12 vCPUs (see section 8), so test `OMP_NTHREAD 12` vs 8 (small gain at best) and
   look for a GPU pod with more vCPUs per GPU for the CPU-bound wave levels. Consider a build with `--timing_solver=true` to split `Flu_Adv` into prepare/GPU/close.
   Use `OPT__OUTPUT_TOTAL` intermediate dumps so a restart is possible (this run had none).
4. Density-refinement run per section 7b (timed, with intermediate dumps and cleanup loop); agree thresholds with the user first.
5. FD4 test in LSS_Hybrid: same ICs with `POT_STENCIL_ORDER 2` vs `4` (add the line after `SOR_MIN_ITER`), `OPT__POT_INT_SCHEME`/`OPT__RHO_INT_SCHEME` stay 4
   (scheme 5 was worse with FD4 in the unit test), compare the halo profiles. See `FD4_POISSON_PLAN.md` section 6 for what is and is not established.
6. Offline check on the a = 1 snapshot (sub-box FD2 vs FD4 vs spectral solve of its own density), plan step 1.
7. Confirm which "dump 45" the user means (`Data_000045` = a 0.486, table ID 45 = `Data_000048`, a 0.530) and decide the high-res reference (e.g. 128^3 base grid or deeper `MAX_LEVEL`).

Files left on the pod (all small): `/workspace/fd4_scripts/` (the scripts that ran the unit tests, paths inside refer to the old locations), `/workspace/fd4_results/`
(`PotError.txt`, inputs and `pot_debug.txt` of every unit-test run; the big HDF5 outputs were deleted), binaries `gamer_{isolated,hybrid,hybrid_fd4,gravity_fd4,gravity_cpu}_bin`,
`compare_amr.py`/`compare_amr.txt` (IsolatedHalo AMR vs uniform), `lss_heavy/` (inputs and IC only), `soliton_run/`.

Practical notes for the next session (tool quirks hit this session):
- `exec-pod` takes `{id, body:{command:[...]}}`, 25 s max per call, no shell unless `["sh","-c",...]`; long jobs need `nohup ... &` plus a log.
- Never `pkill -f <name>` from `sh -c` with that name in the command line: it kills its own shell (exit 143). Use `pgrep -f '^/bin/bash ./script.sh'` and `kill`.
- When writing C strings or shell code through scripts, a `
` can arrive as a real newline; check diffs (it broke two edits).
- Two background scripts writing into the same directories raced once (`sine_test.sh` and `sine_cases.sh`); kill the old one first.
- GAMER's `Flag_Rho` thresholds in `Hydro/Gravity` are x4 per level; a large central density exceeds `MAX_PATCH` (1e6) and aborts.
- Working copy in this repo (`runpod/`, `toy_models/`) is uncommitted. The GAMER fork branch `fd4-poisson` is pushed.

## Not recorded here

Pod credentials. The Jupyter password and any API key stay in the Runpod console.
