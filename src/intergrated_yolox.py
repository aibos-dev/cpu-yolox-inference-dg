import argparse
import os
import time
from loguru import logger
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import psutil
from typing import Dict, List, Tuple, Optional
import logging
import warnings
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import traceback
import torchvision

# Import COCO classes
from yolox.data.datasets import COCO_CLASSES

class VideoHandler:
    def __init__(self, video_dir: str):
        self.video_dir = Path(video_dir)
        if not self.video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
        
        self.video_files = list(self.video_dir.glob('*.mp4')) + \
                          list(self.video_dir.glob('*.avi')) + \
                          list(self.video_dir.glob('*.mov'))
        
        if not self.video_files:
            raise FileNotFoundError(f"No video files found in {video_dir}")
            
        logger.info(f"Found {len(self.video_files)} video files")
        
        # Create frames directory
        self.frames_dir = self.video_dir / 'frames'
        self.frames_dir.mkdir(exist_ok=True)
        
        
    def extract_frames(self, target_fps: float = 2.0) -> List[Path]:
        """Extract frames from videos at specified FPS"""
        extracted_frames = []
        
        for video_file in self.video_files:
            logger.info(f"\nProcessing {video_file.name}")
            
            # Create directory for this video's frames
            video_frames_dir = self.frames_dir / video_file.stem
            video_frames_dir.mkdir(exist_ok=True)
            
            cap = cv2.VideoCapture(str(video_file))
            
            # Get video properties
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / original_fps
            
            logger.info(f"Video properties:")
            logger.info(f"- Duration: {duration:.2f} seconds")
            logger.info(f"- Original FPS: {original_fps}")
            logger.info(f"- Total frames: {total_frames}")
            
            # Calculate frame sampling interval
            frame_interval = int(original_fps / target_fps)
            expected_frames = int(duration * target_fps)
            
            logger.info(f"Extracting frames at {target_fps} FPS")
            logger.info(f"Expected number of frames: {expected_frames}")
            
            frame_count = 0
            saved_count = 0
            
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Generate frame filename with timestamp
                    timestamp = frame_count / original_fps
                    frame_file = video_frames_dir / f"frame_{timestamp:.3f}s.jpg"
                    
                    # Save frame
                    cv2.imwrite(str(frame_file), frame)
                    extracted_frames.append(frame_file)
                    saved_count += 1
                    
                    # Show progress
                    if saved_count % 10 == 0:
                        logger.info(f"Saved {saved_count} frames...")
                
                frame_count += 1
            
            cap.release()
            logger.info(f"Completed {video_file.name}: saved {saved_count} frames")
        
        logger.info(f"\nTotal frames extracted: {len(extracted_frames)}")
        return extracted_frames

    def get_frame_paths(self) -> List[Path]:
        """Get paths of all extracted frames"""
        frames = []
        for video_dir in self.frames_dir.iterdir():
            if video_dir.is_dir():
                frames.extend(list(video_dir.glob('*.jpg')))
        return sorted(frames)

class IntegratedPredictor:
    def __init__(
        self,
        model,
        num_classes=80,
        conf_thre=0.3,
        nms_thre=0.3,
        test_size=(640, 640),
        device="cpu",
        fp16=False
    ):
        self.model = model
        self.num_classes = num_classes
        self.conf_thre = conf_thre
        self.nms_thre = nms_thre
        self.test_size = test_size
        self.device = device
        self.fp16 = fp16
        self.cls_names = COCO_CLASSES  # Add COCO classes

        import sys
        yolox_path = Path('C:/Users/SHAMSA/YOLOX')#Path to YOLOX
        if str(yolox_path) not in sys.path:
            sys.path.append(str(yolox_path))
        
        from yolox.data.data_augment import ValTransform
        self.preproc = ValTransform(legacy=False)

    def preprocess(self, img):
        """Preprocess image for inference"""
        if isinstance(img, str):
            img = cv2.imread(img)
        
        
        # Store original image info
        height, width = img.shape[:2]
        img_info = {"height": height, "width": width, "raw_img": img}

        # Calculate resize ratio
        ratio = min(self.test_size[0] / height, self.test_size[1] / width)
        img_info["ratio"] = ratio

        # Use YOLOX preprocessing
        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()

        return img, img_info


    def inference(self, img):
        """Run inference on image"""
        img_tensor, img_info = self.preprocess(img)

        with torch.no_grad():
            try:
                outputs = self.model(img_tensor)
            
                from yolox.utils import postprocess
                
                # Run YOLOX postprocessing
                outputs = postprocess(
                    outputs,
                    self.num_classes,
                    self.conf_thre,
                    self.nms_thre,
                    class_agnostic=True
                )
                
                if outputs[0] is not None:
                    # Scale boxes back to original size
                    outputs[0][:, 0:4] /= img_info["ratio"]
                else:
                    logger.debug("No detections")
                
                return outputs, img_info
                
            except Exception as e:
                logger.error(f"Error during inference: {str(e)}")
                logger.error(traceback.format_exc())
                raise

    def visualize(self, output, img_info, cls_conf=0.35):
        """Visualize detections on image"""
        img = img_info["raw_img"]
        if output is None:
            return img
            
        output = output.cpu()

        bboxes = output[:, 0:4]
        scores = output[:, 4] * output[:, 5]
        cls_ids = output[:, 6]

        # Draw boxes
        vis_img = img.copy()
        for i in range(len(bboxes)):
            box = bboxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            
            if score < cls_conf:
                continue
                
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            # Draw box
            color = (0, 255, 0)  
            cv2.rectangle(vis_img, (x0, y0), (x1, y1), color, 2)
            
            # Draw label
            class_name = self.cls_names[cls_id]
            text = f'{class_name}: {score:.2f}'
            txt_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(vis_img, (x0, y0 - txt_size[1] - 4), (x0 + txt_size[0], y0), color, -1)
            cv2.putText(vis_img, text, (x0, y0 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
        return vis_img

    def postprocess(self, prediction, num_classes, img_info):
        """Postprocess raw model outputs with detailed debugging"""
        outputs = []
        
        for pred in prediction:  # Loop through batch
            try:
                if pred is None:
                    outputs.append(None)
                    continue

                # Debug input prediction
                logger.debug(f"Processing prediction shape: {pred.shape}")
                
                # Get confidence scores
                obj_conf = pred[:, 4]
                class_conf, class_pred = torch.max(pred[:, 5:], 1)
                conf = obj_conf * class_conf
                
                # Debug confidence scores
                logger.debug(f"Max objectness: {obj_conf.max():.4f}")
                logger.debug(f"Max class confidence: {class_conf.max():.4f}")
                logger.debug(f"Max combined confidence: {conf.max():.4f}")

                # Filter by confidence
                mask = conf > self.conf_thre
                pred = pred[mask]
                conf = conf[mask]
                class_pred = class_pred[mask]
                
                logger.debug(f"Predictions after confidence filter: {len(pred)}")

                if len(pred) == 0:
                    outputs.append(None)
                    continue

                # Get boxes
                boxes = pred[:, :4]
                
                # Remove padding and scale boxes
                if "pad" in img_info and "ratio" in img_info:
                    pad = img_info["pad"]
                    ratio = img_info["ratio"]
                    boxes -= torch.cat([pad[0:2], pad[0:2]]).to(boxes)
                    boxes /= ratio
                
                # Apply NMS
                keep = torchvision.ops.nms(boxes, conf, self.nms_thre)
                
                if len(keep) > 0:
                    boxes = boxes[keep]
                    conf = conf[keep]
                    class_pred = class_pred[keep]
                    
                    # Create output tensor
                    output = torch.cat((
                        boxes,
                        conf.unsqueeze(1),
                        class_pred.float().unsqueeze(1)
                    ), 1)
                    
                    logger.debug(f"Final detections: {len(output)}")
                    outputs.append(output)
                else:
                    outputs.append(None)
                    
            except Exception as e:
                logger.error(f"Error in postprocess: {str(e)}")
                logger.error(traceback.format_exc())
                outputs.append(None)

        return outputs

class IntegratedModelEvaluator:
    def __init__(self, video_handler):
        self.device = torch.device("cpu")
        self.models = {}
        self.weights_dir = Path('weights')
        self.video_handler = video_handler
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        
        # YOLO parameters
        self.num_classes = 80
        self.conf_thre = 0.3
        self.nms_thre = 0.3
        self.test_size = (640, 640)

    def load_yolo_model(self, model_name: str, weight_path: Path) -> Optional[torch.nn.Module]:
        """Load YOLO model with integrated approach"""
        try:
            if 'yolox' in model_name.lower():
                # Import YOLOX modules
                import sys

                yolox_path = Path('C:/Users/SHAMSA/YOLOX') #Path to YOLOX
                if not yolox_path.exists():
                    logger.error(f"YOLOX directory not found at {yolox_path}")
                    return None

                if str(yolox_path) not in sys.path:
                    sys.path.append(str(yolox_path))

                from yolox.exp import get_exp
                from yolox.utils import fuse_model

                # Get appropriate exp file
                exp_file = yolox_path / "exps" / "default"
                if 'nano' in model_name.lower():
                    exp_file = exp_file / "yolox_nano.py"
                elif 'tiny' in model_name.lower():
                    exp_file = exp_file / "yolox_tiny.py"
                else:
                    exp_file = exp_file / "yolox_s.py"

                if not exp_file.exists():
                    logger.error(f"Experiment file not found: {exp_file}")
                    return None
                
                sys.path.append(str(exp_file.parent))
                exp = get_exp(str(exp_file))

                model = exp.get_model()
                
                # Load weights
                ckpt = torch.load(weight_path, map_location=self.device)

                if "model" in ckpt:
                    model.load_state_dict(ckpt["model"])
                else:
                    model.load_state_dict(ckpt)
                
                model.to(self.device)
                model.eval()
                
                # Fuse layers for better inference
                model = fuse_model(model)
                
                # Create predictor wrapper
                predictor = IntegratedPredictor(
                        model=model,
                        num_classes=self.num_classes,
                        conf_thre=self.conf_thre,
                        nms_thre=self.nms_thre,
                        test_size=self.test_size,
                        device=str(self.device)
                )
                
                return predictor

            else:
                # Handle any other YOLO variants
                try:
                    from ultralytics import YOLO
                except ImportError:
                    logger.info("Installing ultralytics package...")
                    import subprocess
                    subprocess.check_call(["pip", "install", "ultralytics"])
                    from ultralytics import YOLO
                
                model = YOLO(str(weight_path))
                return model

        except Exception as e:
            logger.error(f"Error loading {model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def evaluate_models(self, num_runs: int = 2):
        """Evaluate models with visualization"""
        results = {}
        frame_paths = self.video_handler.get_frame_paths()

        # Check if we have any frames to process
        if not frame_paths:
            logger.error("No frames available for processing")
            return results

        total_frames = len(frame_paths)
        logger.info(f"\nTotal extracted frames available: {total_frames}")
        

        for model_name, model in self.models.items():
            logger.info(f"\nEvaluating {model_name}...")
            
            processed_frames = 0
            total_detections = 0
            inference_times = []
        
            try:
                # Sample frames evenly
                # Modified sampling logic to ensure we always get some frames
                target_percentage = 0.25  # Process 25% of frames
                min_frames = 40  # Minimum frames to process
                max_frames = 100  # Maximum frames to process
                
                target_frames = min(max_frames, 
                                max(min_frames, 
                                    int(total_frames * target_percentage)))
            
                logger.info(f"Sampling strategy:")
                logger.info(f"- Target percentage: {target_percentage*100}%")
                logger.info(f"- Min frames: {min_frames}")
                logger.info(f"- Max frames: {max_frames}")
                logger.info(f"- Selected frames for processing: {target_frames}")

                # Get evenly spaced indices
                indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
                sample_frames = [frame_paths[i] for i in indices]
                
                # Create output directory for visualizations
                output_dir = Path(f"output/{model_name}")
                output_dir.mkdir(parents=True, exist_ok=True)

                 # Initialize video writer
                first_frame = cv2.imread(str(sample_frames[0]))
                if first_frame is None:
                    logger.error(f"Could not read first frame: {sample_frames[0]}")
                    continue

                height, width = first_frame.shape[:2]
                video_path = output_dir / f"{model_name}_detection.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    str(video_path),
                    fourcc,
                    30.0,  # fps
                    (width, height)
                )
                
                for run in range(num_runs):
                    logger.info(f"Run {run + 1}/{num_runs}")
                    
                    run_start_time = time.time()
                    run_frames = 0

                    for frame_path in sample_frames:
                        frame = cv2.imread(str(frame_path))
                        if frame is None:
                            continue
                    
                        start_time = time.time()
                    
                        try:
                            outputs, img_info = model.inference(frame)
                        
                            if outputs[0] is not None:
                                detections = len(outputs[0])
                                #logger.info(f"Frame {frame_path.name}: {detections} detections")
                                
                                # Visualize detections
                                vis_img = model.visualize(outputs[0], img_info)

                                # Write frame to video
                                video_writer.write(vis_img)
                                
                                # Save individual frame
                                output_path = output_dir / f"{frame_path.stem}_detected.jpg"
                                cv2.imwrite(str(output_path), vis_img)
                            
                            else:
                                detections = 0
                                logger.info(f"Frame {frame_path.name}: No detections")
                                # Write original frame to video when no detections
                                video_writer.write(frame)
                        
                            inference_time = time.time() - start_time
                            inference_times.append(inference_time)
                        
                            total_detections += detections
                            processed_frames += 1
                            run_frames += 1
                        
                        except Exception as e:
                            logger.error(f"Error processing frame {frame_path}: {str(e)}")
                            continue

                    run_time = time.time() - run_start_time
                    run_fps = run_frames / run_time if run_time > 0 else 0
                    logger.info(f"Run {run + 1} completed:")
                    logger.info(f"- Frames processed: {run_frames}")
                    logger.info(f"- Time taken: {run_time:.2f} seconds")
                    logger.info(f"- FPS: {run_fps:.2f}")

                # Release video writer
                video_writer.release()
                logger.info(f"Detection video saved to: {video_path}")
        
                # Calculate metrics
                if inference_times:
                    avg_inference_time = np.mean(inference_times)
                    avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
                    avg_detections = total_detections / total_frames if total_frames > 0 else 0
                    
                    results[model_name] = {
                        'inference_time': avg_inference_time * 1000,
                        'fps': avg_fps,
                        'frames_processed': processed_frames,
                        'total_detections': total_detections,
                        'avg_detections': avg_detections
                    }
        
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                logger.error(traceback.format_exc())

        return results

def main():
    parser = argparse.ArgumentParser("Evaluate YOLO Models on Video")
    parser.add_argument("--video-dir", type=str, required=True, help="directory containing video files")
    parser.add_argument("--weights-dir", type=str, required=True, help="directory containing model weights")
    parser.add_argument("--target-fps", type=float, default=2.0, help="target FPS for frame extraction")
    parser.add_argument("--num-runs", type=int, default=2, help="number of evaluation runs")
    parser.add_argument('--skip-extraction',
                        action='store_true',
                        help='Skip frame extraction if frames already exist')
    args = parser.parse_args()

    # Initialize video handler
    video_handler = VideoHandler(args.video_dir)
    
    # Extract frames if needed
    if not args.skip_extraction or not list(video_handler.frames_dir.glob('*/*.jpg')):
        logger.info(f"Extracting frames at {args.target_fps} FPS...")
        extracted_frames = video_handler.extract_frames(target_fps=args.target_fps)
    else:
        logger.info("Using existing extracted frames...")
        extracted_frames = video_handler.get_frame_paths()
    
    if not extracted_frames:
        logger.error("No frames available for evaluation")
        return
    
    # Initialize evaluator
    evaluator = IntegratedModelEvaluator(video_handler)
    evaluator.weights_dir = Path(args.weights_dir)
    
    # Look for weight files and load corresponding models
    for weight_file in Path(args.weights_dir).glob("*.pth"):
        model_name = weight_file.stem
        try:
            model = evaluator.load_yolo_model(model_name, weight_file)
            if model is not None:
                evaluator.models[model_name] = model
                logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")

    if not evaluator.models:
        raise RuntimeError("No models were successfully loaded")

    # Run evaluation
    results = evaluator.evaluate_models(num_runs=args.num_runs)
    
    # Print results
    logger.info("\nEvaluation Results:")
    for model_name, metrics in results.items():
        logger.info(f"\nModel: {model_name}")
        logger.info(f"Average Inference Time: {metrics['inference_time']:.2f} ms")
        logger.info(f"FPS: {metrics['fps']:.2f}")
        logger.info(f"Frames Processed: {metrics['frames_processed']}")
        logger.info(f"Total Detections: {metrics['total_detections']}")
        logger.info(f"Average Detections per Frame: {metrics['avg_detections']:.2f}")

if __name__ == "__main__":
    main()