"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import sys

import cv2  # Added for video processing
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig


OBJECTS365_CLASSES = [
    (0, 'Person'),
    (1, 'Sneakers'),
    (2, 'Chair'),
    (3, 'Other Shoes'),
    (4, 'Hat'),
    (5, 'Car'),
    (6, 'Lamp'),
    (7, 'Glasses'),
    (8, 'Bottle'),
    (9, 'Desk'),
    (10, 'Cup'),
    (11, 'Street Lights'),
    (12, 'Cabinet/shelf'),
    (13, 'Handbag/Satchel'),
    (14, 'Bracelet'),
    (15, 'Plate'),
    (16, 'Picture/Frame'),
    (17, 'Helmet'),
    (18, 'Book'),
    (19, 'Gloves'),
    (20, 'Storage box'),
    (21, 'Boat'),
    (22, 'Leather Shoes'),
    (23, 'Flower'),
    (24, 'Bench'),
    (25, 'Potted Plant'),
    (26, 'Bowl/Basin'),
    (27, 'Flag'),
    (28, 'Pillow'),
    (29, 'Boots'),
    (30, 'Vase'),
    (31, 'Microphone'),
    (32, 'Necklace'),
    (33, 'Ring'),
    (34, 'SUV'),
    (35, 'Wine Glass'),
    (36, 'Belt'),
    (37, 'Monitor/TV'),
    (38, 'Backpack'),
    (39, 'Umbrella'),
    (40, 'Traffic Light'),
    (41, 'Speaker'),
    (42, 'Watch'),
    (43, 'Tie'),
    (44, 'Trash bin Can'),
    (45, 'Slippers'),
    (46, 'Bicycle'),
    (47, 'Stool'),
    (48, 'Barrel/bucket'),
    (49, 'Van'),
    (50, 'Couch'),
    (51, 'Sandals'),
    (52, 'Basket'),
    (53, 'Drum'),
    (54, 'Pen/Pencil'),
    (55, 'Bus'),
    (56, 'Wild Bird'),
    (57, 'High Heels'),
    (58, 'Motorcycle'),
    (59, 'Guitar'),
    (60, 'Carpet'),
    (61, 'Cell Phone'),
    (62, 'Bread'),
    (63, 'Camera'),
    (64, 'Canned'),
    (65, 'Truck'),
    (66, 'Traffic cone'),
    (67, 'Cymbal'),
    (68, 'Lifesaver'),
    (69, 'Towel'),
    (70, 'Stuffed Toy'),
    (71, 'Candle'),
    (72, 'Sailboat'),
    (73, 'Laptop'),
    (74, 'Awning'),
    (75, 'Bed'),
    (76, 'Faucet'),
    (77, 'Tent'),
    (78, 'Horse'),
    (79, 'Mirror'),
    (80, 'Power outlet'),
    (81, 'Sink'),
    (82, 'Apple'),
    (83, 'Air Conditioner'),
    (84, 'Knife'),
    (85, 'Hockey Stick'),
    (86, 'Paddle'),
    (87, 'Pickup Truck'),
    (88, 'Fork'),
    (89, 'Traffic Sign'),
    (90, 'Balloon'),
    (91, 'Tripod'),
    (92, 'Dog'),
    (93, 'Spoon'),
    (94, 'Clock'),
    (95, 'Pot'),
    (96, 'Cow'),
    (97, 'Cake'),
    (98, 'Dining Table'),
    (99, 'Sheep'),
    (100, 'Hanger'),
    (101, 'Blackboard/Whiteboard'),
    (102, 'Napkin'),
    (103, 'Other Fish'),
    (104, 'Orange/Tangerine'),
    (105, 'Toiletry'),
    (106, 'Keyboard'),
    (107, 'Tomato'),
    (108, 'Lantern'),
    (109, 'Machinery Vehicle'),
    (110, 'Fan'),
    (111, 'Green Vegetables'),
    (112, 'Banana'),
    (113, 'Baseball Glove'),
    (114, 'Airplane'),
    (115, 'Mouse'),
    (116, 'Train'),
    (117, 'Pumpkin'),
    (118, 'Soccer'),
    (119, 'Skiboard'),
    (120, 'Luggage'),
    (121, 'Nightstand'),
    (122, 'Tea pot'),
    (123, 'Telephone'),
    (124, 'Trolley'),
    (125, 'Head Phone'),
    (126, 'Sports Car'),
    (127, 'Stop Sign'),
    (128, 'Dessert'),
    (129, 'Scooter'),
    (130, 'Stroller'),
    (131, 'Crane'),
    (132, 'Remote'),
    (133, 'Refrigerator'),
    (134, 'Oven'),
    (135, 'Lemon'),
    (136, 'Duck'),
    (137, 'Baseball Bat'),
    (138, 'Surveillance Camera'),
    (139, 'Cat'),
    (140, 'Jug'),
    (141, 'Broccoli'),
    (142, 'Piano'),
    (143, 'Pizza'),
    (144, 'Elephant'),
    (145, 'Skateboard'),
    (146, 'Surfboard'),
    (147, 'Gun'),
    (148, 'Skating and Skiing Shoes'),
    (149, 'Gas Stove'),
    (150, 'Donut'),
    (151, 'Bow Tie'),
    (152, 'Carrot'),
    (153, 'Toilet'),
    (154, 'Kite'),
    (155, 'Strawberry'),
    (156, 'Other Balls'),
    (157, 'Shovel'),
    (158, 'Pepper'),
    (159, 'Computer Box'),
    (160, 'Toilet Paper'),
    (161, 'Cleaning Products'),
    (162, 'Chopsticks'),
    (163, 'Microwave'),
    (164, 'Pigeon'),
    (165, 'Baseball'),
    (166, 'Cutting/chopping Board'),
    (167, 'Coffee Table'),
    (168, 'Side Table'),
    (169, 'Scissors'),
    (170, 'Marker'),
    (171, 'Pie'),
    (172, 'Ladder'),
    (173, 'Snowboard'),
    (174, 'Cookies'),
    (175, 'Radiator'),
    (176, 'Fire Hydrant'),
    (177, 'Basketball'),
    (178, 'Zebra'),
    (179, 'Grape'),
    (180, 'Giraffe'),
    (181, 'Potato'),
    (182, 'Sausage'),
    (183, 'Tricycle'),
    (184, 'Violin'),
    (185, 'Egg'),
    (186, 'Fire Extinguisher'),
    (187, 'Candy'),
    (188, 'Fire Truck'),
    (189, 'Billiards'),
    (190, 'Converter'),
    (191, 'Bathtub'),
    (192, 'Wheelchair'),
    (193, 'Golf Club'),
    (194, 'Briefcase'),
    (195, 'Cucumber'),
    (196, 'Cigar/Cigarette'),
    (197, 'Paint Brush'),
    (198, 'Pear'),
    (199, 'Heavy Truck'),
    (200, 'Hamburger'),
    (201, 'Extractor'),
    (202, 'Extension Cord'),
    (203, 'Tong'),
    (204, 'Tennis Racket'),
    (205, 'Folder'),
    (206, 'American Football'),
    (207, 'Earphone'),
    (208, 'Mask'),
    (209, 'Kettle'),
    (210, 'Tennis'),
    (211, 'Ship'),
    (212, 'Swing'),
    (213, 'Coffee Machine'),
    (214, 'Slide'),
    (215, 'Carriage'),
    (216, 'Onion'),
    (217, 'Green Beans'),
    (218, 'Projector'),
    (219, 'Frisbee'),
    (220, 'Washing Machine/Drying Machine'),
    (221, 'Chicken'),
    (222, 'Printer'),
    (223, 'Watermelon'),
    (224, 'Saxophone'),
    (225, 'Tissue'),
    (226, 'Toothbrush'),
    (227, 'Ice Cream'),
    (228, 'Hot Air Balloon'),
    (229, 'Cello'),
    (230, 'French Fries'),
    (231, 'Scale'),
    (232, 'Trophy'),
    (233, 'Cabbage'),
    (234, 'Hot Dog'),
    (235, 'Blender'),
    (236, 'Peach'),
    (237, 'Rice'),
    (238, 'Wallet/Purse'),
    (239, 'Volleyball'),
    (240, 'Deer'),
    (241, 'Goose'),
    (242, 'Tape'),
    (243, 'Tablet'),
    (244, 'Cosmetics'),
    (245, 'Trumpet'),
    (246, 'Pineapple'),
    (247, 'Golf Ball'),
    (248, 'Ambulance'),
    (249, 'Parking Meter'),
    (250, 'Mango'),
    (251, 'Key'),
    (252, 'Hurdle'),
    (253, 'Fishing Rod'),
    (254, 'Medal'),
    (255, 'Flute'),
    (256, 'Brush'),
    (257, 'Penguin'),
    (258, 'Megaphone'),
    (259, 'Corn'),
    (260, 'Lettuce'),
    (261, 'Garlic'),
    (262, 'Swan'),
    (263, 'Helicopter'),
    (264, 'Green Onion'),
    (265, 'Sandwich'),
    (266, 'Nuts'),
    (267, 'Speed Limit Sign'),
    (268, 'Induction Cooker'),
    (269, 'Broom'),
    (270, 'Trombone'),
    (271, 'Plum'),
    (272, 'Rickshaw'),
    (273, 'Goldfish'),
    (274, 'Kiwi Fruit'),
    (275, 'Router/Modem'),
    (276, 'Poker Card'),
    (277, 'Toaster'),
    (278, 'Shrimp'),
    (279, 'Sushi'),
    (280, 'Cheese'),
    (281, 'Notepaper'),
    (282, 'Cherry'),
    (283, 'Pliers'),
    (284, 'CD'),
    (285, 'Pasta'),
    (286, 'Hammer'),
    (287, 'Cue'),
    (288, 'Avocado'),
    (289, 'Hami Melon'),
    (290, 'Flask'),
    (291, 'Mushroom'),
    (292, 'Screwdriver'),
    (293, 'Soap'),
    (294, 'Recorder'),
    (295, 'Bear'),
    (296, 'Eggplant'),
    (297, 'Board Eraser'),
    (298, 'Coconut'),
    (299, 'Tape Measure/Ruler'),
    (300, 'Pig'),
    (301, 'Showerhead'),
    (302, 'Globe'),
    (303, 'Chips'),
    (304, 'Steak'),
    (305, 'Crosswalk Sign'),
    (306, 'Stapler'),
    (307, 'Camel'),
    (308, 'Formula 1'),
    (309, 'Pomegranate'),
    (310, 'Dishwasher'),
    (311, 'Crab'),
    (312, 'Hoverboard'),
    (313, 'Meatball'),
    (314, 'Rice Cooker'),
    (315, 'Tuba'),
    (316, 'Calculator'),
    (317, 'Papaya'),
    (318, 'Antelope'),
    (319, 'Parrot'),
    (320, 'Seal'),
    (321, 'Butterfly'),
    (322, 'Dumbbell'),
    (323, 'Donkey'),
    (324, 'Lion'),
    (325, 'Urinal'),
    (326, 'Dolphin'),
    (327, 'Electric Drill'),
    (328, 'Hair Dryer'),
    (329, 'Egg Tart'),
    (330, 'Jellyfish'),
    (331, 'Treadmill'),
    (332, 'Lighter'),
    (333, 'Grapefruit'),
    (334, 'Game Board'),
    (335, 'Mop'),
    (336, 'Radish'),
    (337, 'Baozi'),
    (338, 'Target'),
    (339, 'French'),
    (340, 'Spring Rolls'),
    (341, 'Monkey'),
    (342, 'Rabbit'),
    (343, 'Pencil Case'),
    (344, 'Yak'),
    (345, 'Red Cabbage'),
    (346, 'Binoculars'),
    (347, 'Asparagus'),
    (348, 'Barbell'),
    (349, 'Scallop'),
    (350, 'Noodles'),
    (351, 'Comb'),
    (352, 'Dumpling'),
    (353, 'Oyster'),
    (354, 'Table Tennis Paddle'),
    (355, 'Cosmetics Brush/Eyeliner Pencil'),
    (356, 'Chainsaw'),
    (357, 'Eraser'),
    (358, 'Lobster'),
    (359, 'Durian'),
    (360, 'Okra'),
    (361, 'Lipstick'),
    (362, 'Cosmetics Mirror'),
    (363, 'Curling'),
    (364, 'Table Tennis')
]

def draw(images, labels, boxes, scores, thrh=0.4):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline="red")
            class_id = lab[j].item()
            class_name = OBJECTS365_CLASSES[class_id-1] if class_id < len(OBJECTS365_CLASSES) else f"Class_{class_id}"
            draw.text(
                (b[0], b[1]),
                text=f"{class_name} {round(scrs[j].item(), 2)}",
                fill="blue",
            )
        print(f"Detected: {class_name}, Confidence: {round(scrs[j].item(), 2)}, Box: {list(b)}")
        im.save("torch_results.jpg")


def process_image(model, device, file_path):
    im_pil = Image.open(file_path).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    output = model(im_data, orig_size)
    labels, boxes, scores = output

    draw([im_pil], labels, boxes, scores)


def process_video(model, device, file_path):
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("torch_results.mp4", fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)

        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        output = model(im_data, orig_size)
        labels, boxes, scores = output

        # Draw detections on the frame
        draw([frame_pil], labels, boxes, scores)

        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'results_video.mp4'.")


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = args.device
    model = Model().to(device)

    # Check if the input file is an image or a video
    file_path = args.input
    if os.path.splitext(file_path)[-1].lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        # Process as image
        process_image(model, device, file_path)
        print("Image processing complete.")
    else:
        # Process as video
        process_video(model, device, file_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-d", "--device", type=str, default="cpu")
    args = parser.parse_args()
    main(args)
