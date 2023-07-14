<img alt="avatar" src="docs/images/banner.jpeg"/>

# MMPose-based CL-Detection 2023 Challenge Baseline

This repository provides a solution based on the MMPose framework,
allowing users to conveniently switch between different components in the workflow.
Additionally, it offers a tutorial on packaging the solution as a Docker image, ensuring that users can upload their own algorithm models for validation on the leaderboard.

**Prizes**: The top three teams will receive a cash prize of 500 euros and a certificate.
Besides, we will invite other winning teams to present their work orally at the MICCAI conference in October 2023.
They will also be listed as co-authors in a TOP journal in the field (MedIA or TMI) for submitting a journal paper.
We welcome everyone to sign up and participate!


**Reproducibility**: Just as we all know the three steps to put an elephant in the refrigerator,
the code in this repository is divided into six steps.
Each step is accompanied by detailed instructions.
You can easily reproduce the results of this repository step by step or customize your own model.
I have already paved the way, so you can dive right in!

**Note**: This repository utilizes the MMPose framework for model construction.
Please ensure that you have understanding of this framework or refer to another repository,
[CL-Detection2023](https://github.com/szuboy/CL-Detection2023),
which is based solely on PyTorch for heatmap regression.
Both repositories include detailed code comments in English and Chinese for your reference.
[[Challenge Website😆](https://cl-detection2023.grand-challenge.org)]
[[Challenge Leaderboard🏆](https://cl-detection2023.grand-challenge.org/evaluation/challenge/leaderboard/)]
[[CHINESE README👀](README.md)]


## Step 1: Installation and Usage

**1. Configure the necessary dependencies:**
Install `MMEngine` and `MMCV` using `MIM` to ensure the smooth installation of the custom-built `MMPose` source code in the project.

```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

**2. Download the code from this repository:** 
To download the code provided in this repository, you can use Git or click on the "`Download ZIP`" button in the top right corner.
After downloading, you can proceed with the `MMPose` installation.

The code provided in this repository includes additional components for the challenge dataset,
such as the `CephalometricDataset` and `CephalometricMetric` classes, to enhance the user experience.
If your current PyTorch version (Pytorch version should 1.8+) is not compatible, you can create a virtual environment using conda for perfect compatibility.
Please refer to the [MMPose Installation Tutorial](https://mmpose.readthedocs.io/en/latest/installation.html#installation) for instructions on setting up a virtual environment.

```
git clone https://github.com/szuboy/CL-Detection2023-MMPose.git
cd CL-Detection2023-MMPose/mmpose_package/mmpose
pip install -r requirements.txt
pip install -v -e .
# "-v" displays more installation-related information
# "-e" installs the package in editable mode, allowing local modifications to take effect without reinstallation
```

**3. Test the installation:** 
To verify that the above installation steps have been executed successfully and that `MMPose` can be used for model training and testing,
you can run the [`step1_test_mmpose.py`](step1_test_mmpose.py) script.
The expected output should correspond to the installed version:

```python
import mmpose
print(mmpose.__version__)
# Expected output: 1.0.0
```


## Step 2: Data Preparation

**1. Obtaining Training Data:**
You should have already downloaded the challenge training images and annotation files.
Please refer to the [detailed instructions](https://cl-detection2023.grand-challenge.org/) on the challenge website.
Specifically, you need to send the completed [registration form](https://drive.google.com/file/d/1wW9W6rkwJmZz9F3rWCNxK0iECbRSb55q/view?usp=sharing) to the corresponding email address to obtain access to the data.
Rest assured that we will promptly review and approve your request. 

**2. Data Preprocessing:**
For the CL-Detection 2023 challenge dataset, the organizers have standardized the images by zero-padding them to a uniform size of `(2400, 2880, 3)`.
This is done to optimize storage expenses and facilitate distribution.
Consequently, some irrelevant regions need to be removed, and JSON files in the required format for the `MMPose` framework need to be generated.
The main purpose of the preprocessing script is to address these two issues.

In this step, you can execute the script [`step2_prepare_coco_dataset.py`](step2_prepare_coco_dataset.py) to automate the aforementioned operations. There are two options for data processing as follows.

- Modify or set the following data access path parameters and result storage paths in the script [`step2_prepare_coco_dataset.py`](step2_prepare_coco_dataset.py), and then run the script:

```python
parser = argparse.ArgumentParser()

# data parameters
parser.add_argument('--mha_file_path', type=str, default='/data/zhangHY/CL-Detection2023/train_stack.mha')
parser.add_argument('--train_gt_path', type=str, default='/data/zhangHY/CL-Detection2023/train-gt.json')

# save processed images dir path
parser.add_argument('--image_save_dir', type=str, default='/data/zhangHY/CL-Detection2023/MMPose')

experiment_config = parser.parse_args()
main(experiment_config)
```

- Alternatively, you can pass the parameters through the command line in the terminal to run the script. Here's the command you can use:

```
python step2_prepare_coco_dataset.py \
--mha_file_path='/data/zhangHY/CL-Detection2023/train_stack.mha' \
--train_gt_path='/data/zhangHY/CL-Detection2023/train-gt.json' \
--image_save_dir='/data/zhangHY/CL-Detection2023/MMPose'
```

After running the code, you will have a folder containing the images without zero-padding and three JSON files.
This repository follows a **train-validate-test** approach, where the model is trained on the training set, model training and hyperparameter selection are performed on the validation set, and the final model performance is evaluated on the test set.

The current data split logic randomly divides the 400 images as follows: 300 images for training and 50 images each for the validation and test sets. However, you are not bound to this specific partitioning scheme. You can modify it according to your requirements. For example, you can choose to only have a training and validation set, allowing for more training images, which may improve the model's performance.

**Note:**
In order to meet the requirements of the MMPose framework,
we need to convert the training image data and its corresponding annotation files into a format similar to COCO format, which will then be parsed.
The only difference is that we have added the `spacing` key in the `image` information.
This represents the pixel spacing and serves as a "scale" factor for converting pixel distances to physical distances, ensuring accurate metric calculations in subsequent steps.

Additionally, you may have noticed, or perhaps you haven't,
but it's important to point out that the `bbox` coordinates in the generated JSON data files currently default to the width and height of the image.
This setting is based on two main consideration:

- **Compatibility with the MMPose framework:**
During the challenge testing phase, only the input image matrix will be provided, without any additional information.
To adapt to the `TopdownPoseEstimator` detector in the MMPose framework, the default approach is to use the entire image size as the bounding box.
This allows for faster experimentation and optimization using various modules.

- **Flexibility for customization:**
According to a [research paper](https://www.sciencedirect.com/science/article/pii/S1361841520302681) published in the MedIA journal in 2021, it has been demonstrated that a cascade approach can achieve better results.
This gives you more flexibility when optimizing your model. For example, you can start with a detection network to roughly identify the keypoint regions before performing more precise localization.
Employing a cascade approach can be a powerful technique for improving performance.


```
"images": [
    {
        "id": 104,
        "file_name": "104.png",
        "width": 1935,
        "height": 2400,
        "spacing": 0.1
    }
    ...
]

"annotations": [
    {
        "id": 104,
        "image_id": 104,
        "category_id": 1,
        "keypoints": [...],
        "iscrowd": 0,
        "bbox": [
            0,
            0,
            1935,
            2400
        ],
        "area": 4644000
    },
    ...
]
```

Congratulations on reaching this point! You've already accomplished half of the journey.
At this stage, you have the image data and the properly configured files that comply with MMPose requirements.
Now, it's your time to shine! You can freely explore and optimize using the MMPose framework, aiming to secure a place on the leaderboard.🔥


## Model Training


In the `cldetection_configs` folder,
I have provided a complete configuration file for a baseline model based on `HRNet`.
Almost all the configuration options are available for you to customize according to your needs.
You can seamlessly switch and operate using your experience with MMPose.

Just a friendly reminder: don't forget to modify the data root directory in the configuration file.
Currently, the paths in the configuration file are set to the author's own server configuration.
Please make sure to modify them to correspond to the data paths on your platform.

```
dataset_type = 'CephalometricDataset'
data_mode = 'topdown'
data_root = '/data/zhangHY/CL-Detection2023'  # Don't forget to modify this data root directory
```

 Indeed, the provided `step3_train_and_evaluation.py` file is essentially the `tools/train.py` file from MMPose,
 with a renamed filename for better clarity.
 Therefore, you can directly use it for training and evaluation:

```
CUDA_VISIBLE_DEVICES=0 python step3_train_and_evaluation.py \
cldetection_configs/td-hm_hrnet-w32_udp-8xb64-250e-512x512_KeypointMSELoss.py \
--work-dir='/data/zhangHY/CL-Detection2023/MMPose-checkpoints' 
```

After executing the command, you may see output information similar to the following,
along with evaluation metrics of the model on the validation dataset.
The corresponding model weights will be saved in the specified `--work-dir` directory.
You can navigate to that directory to find the corresponding weight files.

```
06/28 03:16:23 - mmengine - INFO - Epoch(train)   [1][ 1/38]  lr: 5.000000e-07  eta: 15:19:03  time: 5.805217  data_time: 1.997265  memory: 8801  loss: 0.010580  loss_kpt: 0.010580  acc_pose: 0.000000
06/28 03:16:24 - mmengine - INFO - Epoch(train)   [1][ 2/38]  lr: 1.501002e-06  eta: 8:38:45  time: 3.277006  data_time: 1.012852  memory: 9545  loss: 0.010586  loss_kpt: 0.010586  acc_pose: 0.003289
06/28 03:16:24 - mmengine - INFO - Epoch(train)   [1][ 3/38]  lr: 2.502004e-06  eta: 6:23:50  time: 2.425048  data_time: 0.683300  memory: 9545  loss: 0.010585  loss_kpt: 0.010585  acc_pose: 0.000000
...
06/28 03:17:27 - mmengine - INFO - Epoch(train)   [2][38/38]  lr: 7.557515e-05  eta: 2:24:11  time: 0.868173  data_time: 0.190205  memory: 5290  loss: 0.008559  loss_kpt: 0.008559  acc_pose: 0.960526
06/28 03:17:27 - mmengine - INFO - Saving checkpoint at 2 epochs
06/28 03:17:35 - mmengine - INFO - Epoch(val)   [2][ 1/13]    eta: 0:00:17  time: 1.422160  data_time: 1.269009  memory: 1233  
06/28 03:17:35 - mmengine - INFO - Epoch(val)   [2][ 2/13]    eta: 0:00:08  time: 0.775162  data_time: 0.637363  memory: 1233
...
06/28 03:17:39 - mmengine - INFO - Evaluating CephalometricMetric...
=> Mean Radial Error        :  MRE = 5.017 ± 6.976 mm
=> Success Detection Rate   :  SDR 2.0mm = 19.211% | SDR 2.5mm = 28.789% | SDR 3mm = 39.000% | SDR 4mm = 56.737%
06/28 03:17:39 - mmengine - INFO - Epoch(val) [2][13/13]  MRE: 5.017320  SDR 2.0mm: 19.210526  SDR 2.5mm: 28.789474  SDR 3.0mm: 39.000000  SDR 4.0mm: 56.736842data_time: 0.274684  time: 0.42305
```

If you want to learn more about additional configuration options, I recommend referring to the corresponding [MMPose documentation](https://mmpose.readthedocs.io/en/latest/user_guides/train_and_test.html#train-with-your-pc).
You can also feel free to raise any questions in the [Issues](https://github.com/szuboy/CL-Detection2023-MMPose/issues) section of this repository.


**Notes:**
These are areas that require careful modification to avoid potential pitfalls.
It is essential to have a solid understanding of the underlying principles before making any modifications.
Please refrain from making changes if you are unsure about the implications:

**1. Model Weight Saving Criteria:**
By default, the model checkpoints are saved based on the `SDR 2.0mm` evaluation metric.
This is a custom evaluation metric specific to this repository and is not available in the official MMPose.
However, it is included in the modified version of this repository. You can safely utilize it. The corresponding configuration code is as follows:

```python
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=2,
        save_best='SDR 2.0mm',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False))

val_evaluator = dict(
    type='CephalometricMetric')
```

`SDR 2.0mm` is indeed an evaluation metric used in the challenge leaderboard.
Therefore, it can be used as the basis for `save_best`.
However, you also have the option to choose another evaluation metric from the leaderboard, such as `MRE`, `SDR 2.5mm`, `SDR 3.0mm`, or `SDR 4.0mm`, as the criterion for saving the best model.

However, please note that if you choose `MRE` as the criterion,
remember to modify the rule to `rule='less'`, as we would want the error to be minimized.

**2. Compatibility with Data Annotations:**
The CL-Detection2023 challenge provides annotations for 38 keypoints/landmarks for each image,
which are not compatible with the existing data configurations in the MMPose framework.
To save you the hassle of modifying these data configuration settings, the MMPose version in this repository has already been made compatible with the provided annotations. You can use it directly.

```python
dict(type='PackPoseInputs', meta_keys=('id', 'img_id', 'img_path', 'category_id', 'crowd_index', 'ori_shape',
                                       'img_shape', 'input_size', 'input_center', 'input_scale', 'flip',
                                       'flip_direction', 'flip_indices', 'raw_ann_info', 'spacing'))


dataset_type = 'CephalometricDataset'
data_mode = 'topdown'
data_root = '/data/zhangHY/CL-Detection2023'
```

It is crucial to ensure that the `meta_keys` for each `PackPoseInputs` are properly configured,
especially including the necessary keys such as `spacing`. Failing to include these keys may result in errors and prevent the evaluation metrics from being calculated correctly.


## Step 4: Testing and Visualizing Predictions

In this step, you can run the script [`step4_test_and_visualize.py`](step4_test_and_visualize.py) to independently test the trained model and assess its performance.
You can also visualize the predictions of the 38 keypoints on the images.
Specifically, you can perform the following operations in the terminal:

```
CUDA_VISIBLE_DEVICES=0 python step4_test_and_visualize.py \
cldetection_configs/td-hm_hrnet-w32_udp-8xb64-250e-512x512_KeypointMSELoss.py \
'/data/zhangHY/CL-Detection2023/MMPose-checkpoints/best_SDR 2.0mm_epoch_40.pth' \
--show-dir='/data/zhangHY/CL-Detection2023/MMPose-visualize' 
```

After execution, you will be able to observe the performance of the model on the independent test set in terms of various metrics such as `MRE` and `SDR`.
The table below shows the performance results obtained by the author while experimenting with different configurations.
Please note that the repository's code does not fix the random seed, so if your reproduction results have slight deviations from the experimental results provided by the author, it is normal.
Similarly, you can observe the visualizations of some keypoint prediction results in the visualization folder.


| Arch                                                                                                       | Input Size |   MRE (mm)    | SDR 2.0mm (%) | SDR 2.5mm (%) | SDR 3.0mm (%) | SDR 4.0mm (%) |                                               ckpt                                                |                                               log                                               |
|------------------------------------------------------------------------------------------------------------|:----------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [HRNet + AdaptiveWingLoss](cldetection_configs/td-hm_hrnet-w32_udp-8xb64-250e-512x512_AdaptiveWingLoss.py) | 512 × 512  | 2.258 ± 5.935 |    66.000     |    75.421     |    82.737     |    91.000     |    [ckpt](https://drive.google.com/file/d/11zBGGzYpUbpYxyMkDfYcqPnSZuhGxVY2/view?usp=sharing)     |    [log](https://drive.google.com/file/d/1Gw9tObsETbqyM5EoCVTDauQncXjv4lqN/view?usp=sharing)    |
| [HRNet + KeypointMSELoss](cldetection_configs/td-hm_hrnet-w32_udp-8xb64-250e-512x512_KeypointMSELoss.py)              | 512 × 512  | 2.199 ± 4.828 |    65.474     |    75.632     |    82.316     |    90.947     |     [ckpt](https://drive.google.com/file/d/1XA_btR9iGmpxkq-SsQlSefIBK7gCpdTu/view?usp=sharing)    |     [log](https://drive.google.com/file/d/1KNKfWth6w7_jubni6mk0aHW-15vEZvDv/view?usp=sharing)   |
| …                                                                                                          |     …      |       …       |       …       |       …       |       …       |       …       |                                                 …                                                 |                                                …                                                |


## Step 5: Generating `expected_output.json`

The challenge requires the submission of a `Docker` file containing the encapsulated model.
This is to test whether the predictions obtained inside the Docker container match the predictions made by the local model.
To facilitate this, we provide a test file called `stack1.mha`, which is located in the `step6_docker_and_upload/test` folder.
This file contains two images and serves as a simulation of unseen test data. Your task is to load this data and use the trained model to make predictions, saving the results in the required format specified by the challenge.

The format for saving the results, as required by the CL-Detection2023 challenge, is explained below:

```
{
    "name": "Orthodontic landmarks",
    "type": "Multiple points",
    "points": [
        {
            "name": "1",
            "point": [
                831.4453125,
                993.75,
                1
            ]
        },
        {
            "name": "2",
            "point": [
                1473.92578125,
                1035.9375,
                1
            ]
        },
        ...
    ],
    "version": {
        "major": 1,
        "minor": 0
    }
}
```

The predicted results for the keypoints within the `points` list can vary between different models. It is not necessary for them to match the example exactly.
Each element within the `points` prediction list must include two key-value pairs:

- `"name"`: This indicates the index of the predicted keypoint, ranging from 1 to 38, representing the different keypoint categories. There are a total of 38 keypoint categories, each identified by its corresponding index.

- `"point"`: This represents the predicted coordinates of the keypoint.
It consists of three values `[x, y, z]`, where `(x, y)` denotes the coordinates of the keypoint, and `z` indicates the index of the image the keypoint belongs to along the Z-axis of the mha data file. The index starts from 1.

You have two options to generate the `expected_output.json` locally using the trained model.
You can either execute the script [`step5_predict_expected_output.py`](step5_predict_expected_output.py) directly or modify the relevant code if you have made additional modifications to your keypoint localization workflow.

- Modify the parameters in the script [`step5_predict_expected_output.py`](step5_predict_expected_output.py) and run the script directly:

```python
# config file | 模型的配置文件
parser.add_argument('--config_file', type=str, default='./cldetection_configs/td-hm_hrnet-w32_udp-8xb64-250e-512x512_KeypointMSELoss.py')

# data parameters | 数据文件路径和配置文件的路径
parser.add_argument('--load_mha_path', type=str, default='./step5_docker_and_upload/test/stack1.mha')
parser.add_argument('--save_json_path', type=str, default='./step5_docker_and_upload/test/expected_output.json')

# model load dir path | 最好模型的权重文件路径
parser.add_argument('--load_weight_path', type=str, default='/data/zhangHY/CL-Detection2023/MMPose-checkpoints/best_SDR 2.0mm_epoch_40.pth')

# model test hyper-parameters
parser.add_argument('--cuda_id', type=int, default=0)

experiment_config = parser.parse_args()
main(experiment_config)
```

- Set the parameters in the terminal and use command-line arguments to run the script:

```
python step5_predict_expected_output.py \
--config_file='./cldetection_configs/td-hm_hrnet-w32_udp-8xb64-250e-512x512_KeypointMSELoss.py' \
--load_mha_path='./step5_docker_and_upload/test/stack1.mha' \
--save_json_path='./step5_docker_and_upload/test/expected_output.json' \
--load_weight_path='/data/zhangHY/CL-Detection2023/MMPose-checkpoints/best_SDR 2.0mm_epoch_40.pth'
--cuda_id=0
```

Since `stack1.mha` contains only two test images, the script will run quickly and generate the expected output file `expected_output.json`.
It's important to note that the `expected_output.json` will differ for different model algorithms.
If you want to test your own model algorithm, you must re-run the script to generate the expected output specific to your model.


## Step 6: Packaging as Docker and Uploading to the Grand Challenge

First, make sure that you have installed `Docker` and `NVIDIA Container Toolkit` on your local computing platform.
These are two important dependencies for the algorithm packaging process.
The former is required for the packaging itself, and the latter ensures that Docker can utilize the GPU.
It is essential to have these dependencies properly installed and configured on your system.

Next, pay attention to the modifications required in the `requirements.txt` file.
Update the project's relevant dependencies in the file (Note: the `torch` module is already included in the pulled image, so there's no need to install it again).
Ensure that all the necessary dependencies for the prediction process are listed in the requirements.txt file to ensure the correct execution of the prediction code and obtain the desired results.

Then, copy the entire `mmpose_package` folder, the model's weight file (without spaces or decimals),
and the `cldetection_utils.py` utility file to the `step5_docker_and_upload` directory.
This ensures that the MMPose framework can be installed through the `Dockerfile` and that the model weights can be successfully loaded.
Additionally, don't forget to copy the `expected_output.json` file generated by the `step5_predict_expected_output.py` script to the test folder.
The final folder structure should look as follows:

```
│  test.sh
│  Dockerfile
│  build.sh
|  best_model_weight.pth
│  process.py
│  requirements.txt
│  export.sh
│  .dockerignore
|  cldetection_utils.py
|  td-hm_hrnet-w32_udp-8xb64-250e-512x512_KeypointMSELoss.py
│
├─test
│      stack1.mha
│      expected_output.json
│
└─mmpose_package
│      mmpose
│            ...
|
```

 If you need to copy additional files or folders, or if you have updated file names,
 please modify the following code accordingly. However, make sure to copy them to the `/opt/algorithm/` directory. 

```dockerfile
COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm mmpose_package /opt/algorithm/mmpose_package
COPY --chown=algorithm:algorithm cldetection_utils.py /opt/algorithm/
COPY --chown=algorithm:algorithm best_model_weight.pth /opt/algorithm/
COPY --chown=algorithm:algorithm td-hm_hrnet-w32_udp-8xb64-250e-512x512_KeypointMSELoss.py /opt/algorithm/
``` 

And then, in the `process.py` file, you can implement your algorithm's inference and testing process within the `predict()` function.
Modify the `save()` function based on the return value of your `predict()` function.

Next, execute the `build.sh` script using the command `sudo ./build.sh`.
Please refrain from modifying any code within the script. This step will build the Docker image and check for any errors that may need troubleshooting.
If everything goes smoothly, you may see a similar output to the following:

```
[+] Building 298.7s (5/16)                                                                                       
 => [internal] load build definition from Dockerfile                                                        0.0s 
 => => transferring dockerfile: 4.07kB                                                                      0.0s 
 => [internal] load .dockerignore                                                                           0.0s
 => => transferring context: 61B                                                                            0.0s
 => [internal] load metadata for docker.io/pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel                      3.2s
 => CANCELED [ 1/12] FROM docker.io/pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel@sha256:ed167cae955fa654c  295.5s
 => => resolve docker.io/pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel@sha256:ed167cae955fa654cefc3663fc0c7b  0.0s
...
...
 => => naming to docker.io/library/cldetection_alg_2023                                                     0.0s
```

Lastly, execute the `sudo ./test.sh` script to verify whether the Docker output matches the locally predicted results. If the results match, you will see the message "Tests successfully passed..." indicating that the local model's predictions are consistent with the results obtained from Docker. 

```
    ...
        }
    ],
    "type": "Multiple points",
    "version": {
        "major": 1,
        "minor": 0
    }
}
Tests successfully passed...
cldetection_alg_2023-output-b35388ee544f2a598b5fb5b088494e5c
```


Finally, directly execute the `sudo ./export.sh` script to export the Docker file that can be uploaded to the challenge platform.
The exported Docker file will be named `CLdetection_Alg_2023.tar.gz`. You can then create an algorithm page on the Grand Challenge platform and submit your algorithm (progress 100%, congratulations 🌷).


## Tips for Participants

This repository only provides a baseline model and a complete workflow for training, testing, and packaging for participants.
The performance of the model is not very high,
and the organizers may suggest the following directions for optimization as a reference:

- Design preprocessing and data augmentation strategies that are more targeted. This repository only involves simple image scaling to a size of `(512, 512)` and horizontal flipping for augmentation.
- Replace the backbone network with more powerful models such as the `HRNet`, `Hourglass` models, or `Transformer` models with self-attention mechanisms.
- Incorporate powerful attention modules. It is common in research to enhance model generalization and performance using attention mechanisms.
- Choosing a suitable loss function can make it easier for the deep learning model to learn and converge more quickly, leading to higher performance.

If you encounter any challenges or difficulties while participating in the CL-Detection 2023 challenge,
encounter any errors while running the code in this repository,
or have any suggestions for improving the baseline model, please feel free to raise an issue.
I will be actively available to provide assistance and support.