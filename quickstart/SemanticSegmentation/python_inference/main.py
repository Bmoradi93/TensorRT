# !python3 -c 'import tensorrt; print("TensorRT version: {}".format(tensorrt.__version__))'
from warnings import catch_warnings
import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from sensor_msgs.msg import Image as img
from sensor_msgs.msg import PointCloud2
import rosbag
import cv2
import ros_numpy

import matplotlib.pyplot as plt
from PIL import Image
import time
from cv_bridge import CvBridge, CvBridgeError
from math import nan, isnan

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs

"""
Author: Behnam Moradi
Email: behnammoradi026@gmail.com
"""

class myClass:
    def __init__(self):
        self.TRT_LOGGER = trt.Logger()
        self.engine_file = "m5.engine"
        self.rosbag_path = "test_bag.bag"
        self.bridge = CvBridge()
        self.engine = self.load_engine(self.engine_file)
        self.dim = (349, 190)
    
    def main(self):
        bag = rosbag.Bag(self.rosbag_path)
        for topic, msg, t in bag.read_messages(topics=['/camera/color/image_raw', '/camera/depth/points']):
            points = self.get_points_array(topic, msg)

            try:
                hand_mask, object_mask, mask = self.get_segmentation_map(topic, msg)
            except:
                print("Something is wrong with the mask generation process")

            # hand_edge_img = self.canny_edge_detection(hand_mask)
            # object_edge_img = self.canny_edge_detection(object_mask)
            # neighboor_pixels = self.get_neighbor_pixels(hand_edge_img, object_edge_img)
            # print(neighboor_pixels)


            # if topic == '/camera/depth/points':
                # self.get_edge_points(object_edge_img, points)
                # self.get_neighbor_points(neighboor_pixels, points)
                # self.plot_contact_lines(neighboor_pixels, points)
        bag.close()
    
    def get_points_array(self, topic, msg):
        if topic == '/camera/depth/points':
            xyz_points = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg, remove_nans=False)
            # print("Point cloud is loaded successfully: " + "Shape: " + str(xyz_points.shape))
            return xyz_points[66:256,107:456,:]
    
    def get_segmentation_map(self, topic, msg):
        mean = np.array([0.485, 0.456, 0.406]).astype('float32')
        stddev = np.array([0.229, 0.224, 0.225]).astype('float32')
        if topic == '/camera/color/image_raw':
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            print(cv_image.shape)
            data = (cv_image.astype('float32') / float(255.0) - mean) / stddev
            input_image =  np.moveaxis(data, 2, 0)
            try:
                mask_image = self.infer(self.engine, input_image)
            except:
                print("Image is not readable!")
            cv_image[:,:,0] = np.where(mask_image == 1, 255/2, mask_image)
            cv_image[:,:,1] = np.where(mask_image == 2, 255/2, mask_image)

            hand_mask = cv_image.copy()
            hand_mask[:,:,0] = np.where(mask_image == 2, 255/2, mask_image)
            hand_mask[:,:,1] = 0
            hand_mask[:,:,2] = 0
            hand_mask = cv2.resize(hand_mask, self.dim, interpolation = cv2.INTER_AREA)

            object_mask = cv_image.copy()
            object_mask[:,:,2] = np.where(mask_image == 1, 255/2, mask_image)
            object_mask[:,:,0] = 0
            object_mask[:,:,1] = 0
            object_mask = cv2.resize(object_mask, self.dim, interpolation = cv2.INTER_AREA)

            mask = cv_image.copy()
            mask[:,:,2] = np.where(mask_image == 1, 255/2, mask_image)
            mask[:,:,0] = np.where(mask_image == 2, 255/2, mask_image)
            mask[:,:,1] = 0
            mask = cv2.resize(mask, self.dim, interpolation = cv2.INTER_AREA)
            

            cv2.imshow("image", mask)
            cv2.waitKey(1)
            return hand_mask, object_mask, mask
    
    def canny_edge_detection(self, img):
        edges_img = cv2.Canny(image=img, threshold1=100, threshold2=200)
        edges_img = self.get_fatten_edge(edges_img, 1)
        # cv2.imshow('Canny_edge_detector', edges_img)
        # cv2.waitKey(1000)
        return edges_img
    
    def get_fatten_edge(self, edge_img, fatten_number):
        fatten_edge_img = np.zeros(edge_img.shape)
        x_dim, y_dim = edge_img.shape
        for i in range(0, x_dim):
            for j in range(0, y_dim):
                if edge_img[i, j] > 0:
                    fatten_edge_img[i-fatten_number:i+fatten_number, j-fatten_number:j+fatten_number] = 1.0
        return fatten_edge_img
    
    def get_edge_points(self, edge_img, points):
        # points = load_point_cloud(point_cloud_path)
        edge_points = np.zeros(points.shape)
        edge_points[:,:,0] = np.multiply(edge_img, points[:,:,0])
        edge_points[:,:,1] = np.multiply(edge_img, points[:,:,1])
        edge_points[:,:,2] = np.multiply(edge_img, points[:,:,2])
        # cv2.imshow('Edge_image', edge_points)
        # cv2.waitKey(0)
        return edge_points
    
    def get_neighbor_pixels(self, hand_edge, object_edge):
        hand_i, hand_j = np.where(hand_edge == 1.0)
        obj_i, obj_j = np.where(object_edge == 1.0)
        neighbors = []
        for i in range(0, len(hand_i)):
            for j in range(0, len(obj_i)):
                d = np.sqrt((hand_i[i] - obj_i[j])**2 + (hand_j[i] - obj_j[j])**2)
                if d < 2:
                    neighbors.append([hand_i[i], hand_j[i], obj_i[j], obj_j[j]]) # [hand_x, hand_y, object_x, object_y] pixel locations
        # print(neighbors)
        return neighbors
    
    def get_neighbor_points(self, neighbors, points):
        hand_neighbor_points = np.zeros(points.shape)
        object_neighbor_points = np.zeros(points.shape)
        (a1, a2, a3) = points.shape
        final_image = np.zeros((a1, a2))
        neighbors_points_eucidean_distance = []
        for p in neighbors:
            if 0.0 not in [points[p[0], p[1], 0], points[p[2], p[3], 0]]:
                hand_neighbor_points[p[0], p[1], :] = points[p[0], p[1], :]
                object_neighbor_points[p[2], p[3], :] = points[p[2], p[3], :]
                d = np.sqrt((points[p[0], p[1], 0] - points[p[2], p[3], 0])**2 + (points[p[0], p[1], 1] - points[p[2], p[3], 1])**2 + (points[p[0], p[1], 2] - points[p[2], p[3], 2])**2)
                final_image[p[0], p[1]] = d
                neighbors_points_eucidean_distance.append(d)
        neighbors_points_eucidean_distance = [x for x in neighbors_points_eucidean_distance if isnan(x) == False]
        # print(max(neighbors_points_eucidean_distance))
        neighbor_points_total = object_neighbor_points - hand_neighbor_points
        # cv2.namedWindow('object_neighbor_points',cv2.WINDOW_NORMAL)
        cv2.imshow('object_neighbor_points', neighbor_points_total)
        # cv2.imshow('object_neighbor_points', final_image)
        # np.savetxt("xyz_points.txt", xyz_points_reshaped)
        cv2.waitKey()
        print("Contact Probability: " + str(sum(neighbors_points_eucidean_distance)/len(neighbors_points_eucidean_distance)))
        # print(neighbors_points_eucidean_distance)
        plt.plot(neighbors_points_eucidean_distance)
        plt.show()
        return final_image

    def postprocess(self, data):
        num_classes = 21
        # create a color palette, selecting a color for each class
        palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = np.array([palette*i%255 for i in range(num_classes)]).astype("uint8")
        # plot the segmentation predictions for 21 classes in different colors
        img = Image.fromarray(data.astype('uint8'), mode='P')
        img.putpalette(colors)
        return img

    def load_engine(self, engine_file_path):
        assert os.path.exists(engine_file_path)
        # print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())


    def infer(self, engine, input_image):
        # print("Reading input image from file")
        image_width = 1280
        image_height = 720

        with engine.create_execution_context() as context:
            # Set input shape based on image dimensions for inference
            context.set_binding_shape(engine.get_binding_index("input"), (1, 3, image_height, image_width))
            # Allocate host and device buffers
            bindings = []
            for binding in engine:
                binding_idx = engine.get_binding_index(binding)
                size = trt.volume(context.get_binding_shape(binding_idx))
                dtype = trt.nptype(engine.get_binding_dtype(binding))
                if engine.binding_is_input(binding):
                    input_buffer = np.ascontiguousarray(input_image)
                    input_memory = cuda.mem_alloc(input_image.nbytes)
                    bindings.append(int(input_memory))
                else:
                    output_buffer = cuda.pagelocked_empty(size, dtype)
                    output_memory = cuda.mem_alloc(output_buffer.nbytes)
                    bindings.append(int(output_memory))

            stream = cuda.Stream()
            # Transfer input data to the GPU.
            cuda.memcpy_htod_async(input_memory, input_buffer, stream)
            # Run inference
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            # Transfer prediction output from the GPU.
            cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
            # Synchronize the stream
            stream.synchronize()
            segmented_mask = np.reshape(output_buffer, (image_height, image_width))
            # np.savetxt("segmap.txt", segmented_mask)
            # print(input_image[1,:,:].shape)
            # print(segmented_mask.shape)
            return segmented_mask

    def plot_contact_lines(self, neighbors, points):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for p in neighbors:
            if 0.0 not in [points[p[0], p[1], 0], points[p[2], p[3], 0]]:
                point_z = [points[p[0], p[1], 0], points[p[2], p[3], 0]]
                point_y = [points[p[0], p[1], 1], points[p[2], p[3], 1]]
                point_x = [points[p[0], p[1], 2], points[p[2], p[3], 2]]
                ax.plot(point_x, point_y, point_z)
        plt.show()

if __name__ == "__main__":   
    # main(rosbag_path)
    my_class = myClass()
    my_class.main()
    # my_class.canny_edge_detection('output.jpg')

