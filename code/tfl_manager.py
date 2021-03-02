import pickle
from utils import Color
from visualize_test import visualize
import find_tfl_lights
import cv2
import numpy as np
import SFM
from frame_container import FrameContainer
from create_sample import crop_image
from tensorflow.keras.models import load_model


class TFLMan:

    def __init__(self, pkl_path):
        with open(pkl_path, 'rb') as pklfile:
            self.__pkl_data = pickle.load(pklfile, encoding='latin1')

        self.__pp = self.__pkl_data['principle_point']
        self.__focal = self.__pkl_data['flx']
        self.__prev_frame = None
        self.__model = load_model("data/model.h5")

    def on_frame(self, current_frame, frame_index):
        # phase 1
        candidates, auxliary = self.__get_candidates(current_frame)
        assert len(candidates) == len(auxliary)
        assert len(candidates) >= 0        
        
        # phase 2
        traffic_lights, traffic_auxiliary = self.__get_tfl_coordinates(current_frame, candidates, auxliary)
        if len(traffic_lights) > len(candidates):
            traffic_lights = candidates
        assert len(traffic_lights) == len(traffic_auxiliary)
        assert len(traffic_lights) >= 0

        # phase 3
        current_frame = FrameContainer(current_frame)
        current_frame.traffic_light = traffic_lights

        if self.__prev_frame:
            try:
                current_frame.EM = self.__pkl_data['egomotion_' + str(frame_index - 1) + '-' + str(frame_index)]
            except KeyError:
                pass # I have not yet decided how to handle this case

            distance = self.__get_dists(self.__prev_frame, current_frame, self.__prev_frame.traffic_light, traffic_lights)
            assert len(distance) == len(traffic_lights)
        else:
            distance = None
        
        self.__prev_frame = current_frame

        visualize(current_frame, candidates, auxliary, traffic_lights, traffic_auxiliary, distance)

        return traffic_lights, traffic_auxiliary, distance

    @staticmethod
    def __get_candidates(image):
        x_red, y_red, x_green, y_green = find_tfl_lights.find_tfl_lights(cv2.imread(image))
        assert len(x_red) == len(y_red)
        assert len(x_green) == len(y_green)
        
        candidates = [[x_red[index], y_red[index]] for index in range(len(x_red))] + [[x_green[index], y_green[index]] for index in range(len(x_green))]
        auxliary = [Color.red for i in x_red] + [Color.green for i in x_green]

        return candidates, auxliary

    def __get_tfl_coordinates(self, image, candidates, auxliary):

        crop_shape = (81, 81)
        l_predicted_label = []

        for candidate in candidates:
            crop_img = crop_image(cv2.imread(image), candidate[0], candidate[1])
            predictions = self.__model.predict(crop_img.reshape([-1]+list(crop_shape) +[3]))
            l_predicted_label.append(1 if predictions[0][1] > 0.98 else 0)

        traffic_lights = [candidates[index] for index in range(len(candidates)) if l_predicted_label[index] == 1]
        auxliary =  [auxliary[index] for index in range(len(auxliary)) if l_predicted_label[index] == 1]

        return traffic_lights, auxliary

    def __get_dists(self, prev_frame, current_frame, prev_traffic_lights, current_traffic_lights):
        current_frame.traffic_light = np.array(current_traffic_lights)
        curr_frame = SFM.calc_TFL_dist(prev_frame, current_frame, self.__focal, self.__pp)

        return np.array(curr_frame.traffic_lights_3d_location)[:, 2]

