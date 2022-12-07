import sys
import os
sys.path.append(os.path.join('..', 'darknet', 'build', 'darknet', 'x64')) 
import darknet
import cv2

# darknet helper function to run detection on image


def darknet_helper(img, width, height):
    darknet_image = darknet.make_image(width, height, 3)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (width, height),
                             interpolation=cv2.INTER_LINEAR)

    img_height, img_width, _ = img.shape
    width_ratio = img_width / width
    height_ratio = img_height / height

    darknet.copy_image_from_bytes(darknet_image, img_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image)
    darknet.free_image(darknet_image)
    return detections, width_ratio, height_ratio


if __name__ == "__main__":
    network, class_names, class_colors = darknet.load_network("darknet/cfg/yolov4.cfg",
                                                              "darknet/data/obj.data",
                                                              "darknet/satmodel.weights")
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    image = cv2.imread("test1.jpg")

    detections, width_ratio, height_ratio = darknet_helper(image, width, height)

    for label, confidence, bbox in detections:
        left, top, right, bottom = darknet.bbox2points(bbox)
        left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(
            bottom * height_ratio)
        cv2.rectangle(image, (left, top), (right, bottom), class_colors[label], 2)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    class_colors[label], 2)
    cv2.imshow('ex', image)

    cv2.waitKey(0)
