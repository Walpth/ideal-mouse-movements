# Ideal Mouse Movements

Well, maybe not ideal yet

![ideal](ideal.png)

Current pipeline:

```text
log_movements.py
  -> strokes_csv/raw_mouse_data.csv
  -> csv_clean.py
  -> strokes_csv/raw_mouse_cleaned.csv
  -> csv_segment.py
  -> strokes_csv/mouse_segmented.csv
  -> main_pipeline/train.py
  -> main_pipeline/generator.py
```

Plotting scripts:

```text
plot_original.py   -> strokes_original/
plot_segmented.py  -> strokes_segmented/
plot_generated.py  -> strokes_generated/
```
