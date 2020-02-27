import torch
import json

model = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=True)
model.eval()

file_read = open("imagenet_class_index.json").read()
categ = json.loads(file_read)
print(categ['0'])



filename= 'test_0.JPEG'
# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model


with torch.no_grad():
    output = model(input_batch)

# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probab=torch.nn.functional.softmax(output[0], dim=0)


#print(probab.max(0)[0])

idx=sorted(range(len(probab)), key=lambda i: probab[i])[-3:] #top 1 predictions

most_prob=probab[idx[-1]] #probability of most likely category

print('finished')