# Dataset Notes

## SWaT / WADI (Real Data)

**These datasets require a formal access request.** They are NOT freely downloadable.

- Application: https://itrust.sutd.edu.sg/itrust-labs_datasets/
- SWaT: 51 features, water treatment plant, ~11 days of data
- WADI: 123 features, water distribution system

The synthetic versions used in benchmarks (`get_dummy_swat_dataloader`) are
physics-inspired random data that verify implementation correctness only.
For publishable results, use the real SWaT/WADI data after obtaining access.

## ETT Datasets (Free)
```bash
python datasets/download_datasets.py --datasets ett
```

Source: https://github.com/zhouhaoyi/ETDataset

## PSM / SMAP / MSL (Free)
```bash
python datasets/download_datasets.py --datasets psm smap msl
```
