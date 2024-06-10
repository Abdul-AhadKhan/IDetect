import PySimpleGUI as sg
from ultralytics import YOLO
import cv2
import cvzone
import math

classNames = ['Hardhat', 'Mask', 'No Hardhat', 'No Mask', 'No Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'Machinery', 'Vehicle']
unsafe = ['No Hardhat', 'No Mask', 'No Safety Vest']

persons = []
noEquipments = []
machines = []

def detectObject(video, model, conf):

    while True:
        success, img = video.read()

        img = cv2.resize(img, (frame_width, frame_height))

        results = model(img, stream=True)
        if success:
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    confidence = math.ceil((box.conf[0] * 100)) / 100

                    name = int(box.cls[0])

                    if confidence > conf:

                        if classNames[name] in unsafe:
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            noEquipments.append([x1, y1, x2, y2])
                            print("No Equipments: " + f'{noEquipments}')
                            cvzone.putTextRect(img, f'{classNames[name]} {confidence}', (x1, max(30, y1)),
                                               scale=0.7,
                                               thickness=1, colorT=(255, 255, 255), colorR=(0, 0, 255))
                        else:
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                            cvzone.putTextRect(img, f'{classNames[name]} {confidence}', (x1, max(30, y1)),
                                               scale=0.7,
                                               thickness=1)

                        if classNames[name] == "Person":
                            persons.append([x1, y1, x2, y2])
                        if classNames[name] in ['Machinery', 'Vehicle']:
                            machines.append([x1, y1, x2, y2])

                caution = False
                closeToMachine = False
                for person in persons:
                    caution = False
                    closeToMachine = False
                    for equipment in noEquipments:
                        x1, y1, x2, y2 = equipment[0], equipment[1], equipment[2], equipment[3]

                        if x1 > person[0] and y1 > person[1] and x2 < person[2] and y2 < person[3]:
                            caution = True
                            break

                    for machine in machines:
                        x1, y1 = machine[0], machine[1]
                        x2, y2 = machine[2], machine[3]

                        lineLength = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

                        x, y = person[0], person[1]

                        distanceBtwPtnLine = abs((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1)) / lineLength

                        print("Distance: " + f'{distanceBtwPtnLine}')

                        if distanceBtwPtnLine < 100:
                            closeToMachine = True
                            break

                        else:
                            x, y = person[2], person[3]
                            distanceBtwPtnLine = abs((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1)) / lineLength
                            print("Distance: " + f'{distanceBtwPtnLine}')
                            if distanceBtwPtnLine < 100:
                                closeToMachine = True
                                break

                    x1 = person[0]
                    y1 = person[1]
                    x2 = person[2]
                    y2 = person[3]
                    if caution:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cvzone.putTextRect(img, f'{"Caution!"}', (x1, max(30, y1)), scale=1, thickness=1,
                                           colorT=(255, 255, 255), colorR=(0, 0, 255))

                    if closeToMachine:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cvzone.putTextRect(img, f'{"Close To Machine!"}', (x1, max(30, y1)), scale=1, thickness=1,
                                           colorT=(255, 255, 255), colorR=(0, 0, 255))

                persons.clear()
                noEquipments.clear()
                machines.clear()

            cv2.imshow("Image", img)
            key = cv2.waitKey(1)

            if ord('p') == key:
                cv2.waitKey(-1)

            if key == 27:
                break

    video.release()
    cv2.destroyAllWindows()


frame_width = 1280
frame_height = 720

layout = [
    [sg.Text('GUI Object Detection with Yolo V8')],
    [sg.Text('Choose Video File'), sg.InputText(key='video_file', enable_events=True), sg.FileBrowse()],
    [sg.Text('Select Model'), sg.Combo(['nano', 'medium', 'large'], default_value='nano', key='model')],
    [sg.Text('Minimum Confidence'), sg.InputText('0.5', key='min_confidence')],
    [sg.Button('Run'), sg.Button('Stop'), sg.Button('Open Webcam')],
    [sg.Image(filename='', key='image')]
]


# Create the Window
window = sg.Window('GUIYoloV8', layout, finalize=True)
run_model = False
video = None

# Event Loop to process "events"
while True:
    event, values = window.read(timeout=1)

    # When a video file is chosen
    if event == 'video_file':
        video_file = values['video_file']
        if video_file:
            video = cv2.VideoCapture(video_file)
            video.set(3, frame_width)
            video.set(4, frame_height)

    # When press Run
    if event == 'Run':

        minimumConfidence = float(values['min_confidence'])
        modelType = values['model']


        modelPath = ''

        if modelType == 'nano':
            modelPath = '../Yolo-Weights/Train_nano_100epochs.pt'
        elif modelType == 'medium':
            modelPath = '../Yolo-Weights/Train_medium_100epochs.pt'
        elif modelType == 'large':
            modelPath = '../Yolo-Weights/Train_large_100epochs.pt'

        if video is not None:

            run_model = True

            model = YOLO(modelPath)

            detectObject(video, model, minimumConfidence)

            # while True:
            #     success, img = video.read()
            #
            #     img = cv2.resize(img, (frame_width, frame_height))
            #
            #     results = model(img, stream=True)
            #     if success:
            #         for result in results:
            #             boxes = result.boxes
            #             for box in boxes:
            #                 x1, y1, x2, y2 = box.xyxy[0]
            #                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #
            #                 confidence = math.ceil((box.conf[0] * 100)) / 100
            #
            #                 name = int(box.cls[0])
            #
            #                 if classNames[name] in unsafe:
            #                     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            #                     noEquipments.append([x1, y1, x2, y2])
            #                     print("No Equipments: " + f'{noEquipments}')
            #                     cvzone.putTextRect(img, f'{classNames[name]} {confidence}', (x1, max(30, y1)),
            #                                        scale=0.7,
            #                                        thickness=1, colorT=(255, 255, 255), colorR=(0, 0, 255))
            #                 else:
            #                     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            #                     cvzone.putTextRect(img, f'{classNames[name]} {confidence}', (x1, max(30, y1)),
            #                                        scale=0.7,
            #                                        thickness=1)
            #
            #                 if classNames[name] == "Person":
            #                     persons.append([x1, y1, x2, y2])
            #                 if classNames[name] in ['Machinery', 'Vehicle']:
            #                     machines.append([x1, y1, x2, y2])
            #
            #         caution = False
            #         closeToMachine = False
            #         for person in persons:
            #             caution = False
            #             closeToMachine = False
            #             for equipment in noEquipments:
            #                 x1, y1, x2, y2 = equipment[0], equipment[1], equipment[2], equipment[3]
            #
            #                 if x1 > person[0] and y1 > person[1] and x2 < person[2] and y2 < person[3]:
            #                     caution = True
            #                     break
            #
            #             for machine in machines:
            #                 x1, y1 = machine[0], machine[1]
            #                 x2, y2 = machine[2], machine[3]
            #
            #                 lineLength = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            #
            #                 x, y = person[0], person[1]
            #
            #                 distanceBtwPtnLine = abs((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1)) / lineLength
            #
            #                 print("Distance: " + f'{distanceBtwPtnLine}')
            #
            #                 if distanceBtwPtnLine < 100:
            #                     closeToMachine = True
            #                     break
            #
            #                 else:
            #                     x, y = person[2], person[3]
            #                     distanceBtwPtnLine = abs((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1)) / lineLength
            #                     print("Distance: " + f'{distanceBtwPtnLine}')
            #                     if distanceBtwPtnLine < 100:
            #                         closeToMachine = True
            #                         break
            #
            #             x1 = person[0]
            #             y1 = person[1]
            #             x2 = person[2]
            #             y2 = person[3]
            #             if caution:
            #                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            #                 cvzone.putTextRect(img, f'{"Caution!"}', (x1, max(30, y1)), scale=1, thickness=1,
            #                                    colorT=(255, 255, 255), colorR=(0, 0, 255))
            #
            #             if closeToMachine:
            #                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            #                 cvzone.putTextRect(img, f'{"Close To Machine!"}', (x1, max(30, y1)), scale=1, thickness=1,
            #                                    colorT=(255, 255, 255), colorR=(0, 0, 255))
            #
            #         persons.clear()
            #         noEquipments.clear()
            #         machines.clear()
            #
            #         cv2.imshow("Image", img)
            #         key = cv2.waitKey(1)
            #
            #         if ord('p') == key:
            #             cv2.waitKey(-1)
            #
            #         if key == 27:
            #             break
            #
            # video.release()
            # cv2.destroyAllWindows()

    elif event == 'Open Webcam':

        video = cv2.VideoCapture(0)
        video.set(3, frame_width)
        video.set(4, frame_height)
        minimumConfidence = float(values['min_confidence'])
        modelType = values['model']

        modelPath = ''

        if modelType == 'nano':
            modelPath = '../Yolo-Weights/Train_nano_100epochs.pt'
        elif modelType == 'medium':
            modelPath = '../Yolo-Weights/Train_medium_100epochs.pt'
        elif modelType == 'large':
            modelPath = '../Yolo-Weights/Train_large_100epochs.pt'

        model = YOLO(modelPath)
        detectObject(video, model, minimumConfidence)
    # When press Stop or close window or press Close
    elif event in ('Stop', sg.WIN_CLOSED, 'Close'):
        if run_model:
            run_model = False  # Stop running
            if video is not None:
                video.release()  # Release video
            window['image'].update(filename='')  # Destroy picture
        # When close window or press Close
        if event in (sg.WIN_CLOSED, 'Close'):
            break




# Close window
window.close()
