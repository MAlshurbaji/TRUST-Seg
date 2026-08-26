# Sample data

This repository contains a small ISLES 2022 sample for checking paths, preprocessing, and file compatibility. It is not the complete ISLES 2022 dataset and must not be used to reproduce the paper's reported results.

The sample contains:

- selected DWI and binary-label PNG slices from `ISLES22_DWI_128`;
- one reconstructed training volume and one test volume;
- matching bbox volumes;
- representative SAM-L, MedSAM2-B, and GrabCut teacher masks;
- an example teacher consensus, uncertainty map, and confidence map.

The sample configuration is `configs/isles22_sample.yaml`. Its one-epoch settings are intended only as a smoke test.

Full datasets and generated checkpoints are ignored by Git. Preserve the same filename conventions when replacing the samples with complete patient-wise train, validation, and test splits.

