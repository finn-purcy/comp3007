import cv2
from ultralytics import YOLO

def thermo_inference(image_path, ptFile):
    yolo = YOLO(ptFile)

    image = cv2.imread(image_path)
    resize = cv2.resize(image, (640, 640), cv2.INTER_LINEAR)


    left_half_image = resize[:,:320]

    results = yolo(left_half_image)

    #keep most centeral detections
    references = [None, None]
    ref_dist = [500, 500]

    print(len(references))
    
    for result in results:
        for box in result.boxes:
            y = int(box.xywh.tolist()[0][1])
        
            dist_from_centre = abs(320 - y)

            if dist_from_centre < ref_dist[0]:
                # shift the current best detection to second place
                references[1] = references[0]
                ref_dist[1] = ref_dist[0]

                references[0] = box

            elif dist_from_centre < ref_dist[1]:
                # better than second place, but not first
                references[1] = box
                ref_dist[1] = dist_from_centre
    

    _, y, _, h = references[0].xywh.tolist()[0]
    a = int(y+h/2)
    A = int(references[0].cls.item()) * 10

    _, y, _, h = references[1].xywh.tolist()[0]
    b = int(y+h/2)
    B = int(references[1].cls.item()) * 10

    w = (B-A)/(a-b)
    c = A+a*((B-A)/(a-b))

    T = c - 320*w    

    print(f"a: {a}\nb: {b}\nA: {A}\n B: {B}")

    print(T)

if __name__ == "__main__":
    thermo_inference('output/task2/thermo1.png/t.png', 'task3.pt')
       