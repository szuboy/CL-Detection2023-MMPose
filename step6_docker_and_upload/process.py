
import SimpleITK
from pathlib import Path
import numpy as np
import torch
import json

from evalutils import DetectionAlgorithm
from evalutils.validators import UniquePathIndicesValidator, UniqueImagesValidator

# 导入 mmpose 使用的相关函数
from mmpose.apis import inference_topdown
from mmpose.structures import merge_data_samples
from mmpose.apis import init_model as init_pose_estimator

# 导入自己的模型结构，比如工具文件
from cldetection_utils import remove_zero_padding


class Cldetection_alg_2023(DetectionAlgorithm):
    def __init__(self):
        # 请不要修改初始化父类的函数，这里的两个路径都不要修改，这是grand-challenge平台的内部路径，不可修改
        super().__init__(
            validators=dict(input_image=(UniqueImagesValidator(), UniquePathIndicesValidator())),
            input_path=Path("/input/images/lateral-dental-x-rays/"),
            output_file=Path("/output/orthodontic-landmarks.json"))

        print("==> Starting...")

        # 使用对应的GPU，注意grand-challenge只有一块GPU，请保证下面的权重加载，加上map_location=self.device设置避免不同设备导致的错误
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载自己的模型和权重文件，这里的权重文件路径是 /opt/algorithm/best_model.pt，
        # 这是因为在docker中会将当前目录挂载为 /opt/algorithm/，所以你要访问当前文件夹的任何文件，在代码中的路径都应该是 /opt/algorithm/
        # 初始化模型，init_pose_estimator 函数内部已经加载了模型权重和开启了eval()模式
        config_file = '/opt/algorithm/td-hm_hrnet-w32_udp-8xb64-250e-512x512.py'
        load_weight_path = '/opt/algorithm/best_model_weight.pth'
        self.pose_estimator = init_pose_estimator(config=config_file, checkpoint=load_weight_path, device=self.device)

        print("==> Using ", self.device)
        print("==> Initializing model")
        print("==> Weights loaded")

    def save(self):
        """TODO: 重写父类函数，根据自己的 predict() 函数返回类型，将结果保存在 self._output_file 中"""

        # 因为我们就只传入了一个文件名，则 self._case_results 列表中只有一个元素，我们仅需要取出来进行解码
        # 这个 all_images_predict_landmarks_list 就是 predict() 函数的返回值
        all_images_predict_landmarks_list = self._case_results[0]

        # 将预测结果调整为挑战赛需要的JSON格式，借助字典的形式作为中间类型
        json_dict = {'name': 'Orthodontic landmarks', 'type': 'Multiple points'}

        all_predict_points_list = []
        for image_id, predict_landmarks in enumerate(all_images_predict_landmarks_list):
            for landmark_id, landmark in enumerate(predict_landmarks):
                points = {'name': str(landmark_id + 1),
                          'point': [landmark[0], landmark[1], image_id + 1]}
                all_predict_points_list.append(points)
        json_dict['points'] = all_predict_points_list

        # 提交的版本信息，可以为自己的提交备注不同的版本记录
        major = 1
        minor = 0
        json_dict['version'] = {'major': major, 'minor': minor}

        # 转为JSON接受的字符串形式
        json_string = json.dumps(json_dict, indent=4)
        with open(str(self._output_file), "w") as f:
            f.write(json_string)

    def process_case(self, *, idx, case):
        """!IMPORTANT: 请不要修改这个函数的任何内容，下面是具体的注释信息"""

        # 调用父类的加载函数，case 这个变量包含当前的堆叠了所有测试图片的文件名，类似如：/.../../test_stack_image.mha
        input_image, input_image_file_path = self._load_input_image(case=case)

        # 传入对应的 input_image SimpleITK.Image 格式
        predict_result = self.predict(input_image=input_image)

        # 返回预测结果出去
        return predict_result

    def predict(self, *, input_image: SimpleITK.Image):
        """TODO: 请修改这里的逻辑，执行自己设计的模型预测，返回值可以是任何形式的"""

        # 将 SimpleITK.Image 格式转为 Numpy.ndarray 格式进行处理， stacked_image_array 的形状为 (100, 2400, 2880, 3)
        stacked_image_array = SimpleITK.GetArrayFromImage(input_image)

        # 所有图像的测试结果列表
        all_images_predict_keypoints_list = []

        # 开始测试模型进行测试
        with torch.no_grad():
            self.pose_estimator.eval()
            for i in range(np.shape(stacked_image_array)[0]):
                # 切片出一张图像出来
                image = np.array(stacked_image_array[i, :, :, :])

                # 预处理去除0填充部分
                image = remove_zero_padding(image)

                # 模型调用进行预测，内部已经包含了配置文件中的test_pipeline操作，内部已经进行配置好的预处理操作，直接丢图就好啦
                # 如果前面有一个粗定位的模型，只需要改变bboxes参数，传入检测框坐标即可
                predict_results = inference_topdown(model=self.pose_estimator, img=image, bboxes=None, bbox_format='xyxy')

                # 由于 MMPose 兼容考虑到一张图有多个bboxes，所以返回的结果是多个 bboxes的关键点预测结果，虽然挑战赛的bbox只有一个
                # 但我们还需要调用 merge_data_samples 对结果进行合并
                result_samples = merge_data_samples(predict_results)

                # 取出对应的关键点的预测结果, pred_instances.keypoints shape is (检测框数量，关键点数量，2)，我们就一个检测框，所以索引是0
                keypoints = result_samples.pred_instances.keypoints[0, :, :]

                keypoints_list = []
                for i in range(np.shape(keypoints)[0]):
                    # 索引得到不同的关键点热图
                    x0, y0 = keypoints[i, 0], keypoints[i, 1]
                    keypoints_list.append([x0, y0])
                all_images_predict_keypoints_list.append(keypoints_list)

        print("==========================================")
        print('The prediction is successfully generated!!')
        print("==========================================")

        return all_images_predict_keypoints_list



if __name__ == "__main__":
    algorithm = Cldetection_alg_2023()
    algorithm.process()

    # 问：这里没有实现 process() 函数，怎么可以进行调用呢？
    # 答：因为这是 Cldetection_alg_2023 继承了 DetectionAlgorithm，父类函数，子类也就有了，然后进行执行，背后会自动调用相关函数

    # 问：调用 process() 函数，背后执行了什么操作呢？
    # 答：我们可通过跳转到源码可以看到，process() 函数，这里是源码显示：
    #    def process(self):
    #        self.load()
    #        self.validate()
    #        self.process_cases()
    #        self.save()
    #    我们可以看到背后执行了这四个函数，而对应在 process_cases() 函数中又进行了调用 process_case() 函数：
    #    def process_cases(self, file_loader_key: Optional[str] = None):
    #        if file_loader_key is None:
    #            file_loader_key = self._index_key
    #        self._case_results = []
    #        for idx, case in self._cases[file_loader_key].iterrows():
    #            self._case_results.append(self.process_case(idx=idx, case=case))
    #    因此，对应这我们挑战赛的内容，您仅需要在 process_case() 和 save() 函数中实现你想要的功能

    # 问：又说仅需要 process_case() 和 save() 进行实现，为什么又跳出一个 predict() 函数呢？
    # 答：predict() 函数是父类 DetectionAlgorithm 要求实现的，负责预测每一个case的结果，不然会提示 NotImplementedError 错误



    
