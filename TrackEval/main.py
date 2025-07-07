import os
from trackeval import metrics
from trackeval.eval import Evaluator
from trackeval.datasets.aicity_3d import AICity3D

def run_hota():
    dataset = AICity3D()
    metric = metrics.HOTA3D()

    evaluator = Evaluator({
                    'PRINT_CONFIG': True,
                    'DISPLAY_LESS_PROGRESS': False
                })

    evaluator.evaluate(
        [dataset], [metric]
    )

def convert_to_trackeval_format(input_path, output_path, is_gt=True):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) != 11:
                print(f"Skipping malformed line: {line}")
                continue

            # Unpack parts
            _, _, global_id, frame_id, x, y, z, w, l, h, _ = parts

            frame = int(frame_id)
            obj_id = int(global_id)
            x = float(x)
            y = float(y)
            z = float(z)
            l = float(l)
            w = float(w)
            h = float(h)

            confidence = 1.0 if is_gt else 0.9  # you can adjust as needed

            line_out = f"{frame},{obj_id},{x:.6f},{y:.6f},{z:.6f},{l:.6f},{w:.6f},{h:.6f},{confidence:.6f},-1,-1,-1\n"
            outfile.write(line_out)

    print(f"✅ Converted {input_path} → {output_path}")

def main():
    # INPUT files
    raw_tracker_file = "aicity_25_data/final_track_exp57_fix.txt"
    raw_gt_file = "aicity_25_data/gt.txt"

    # OUTPUT files (TrackEval format)
    out_gt_file = "data/AIC25_Track1/gt/gt.txt"
    out_tracker_file = "data/AIC25_Track1/tracker/exp57/exp57.txt"

    convert_to_trackeval_format(raw_gt_file, out_gt_file, is_gt=True)
    convert_to_trackeval_format(raw_tracker_file, out_tracker_file, is_gt=False)
    run_hota()
    z = 2
if __name__ == "__main__":
    main()