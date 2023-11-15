import torch.nn as nn

class encoder_front:
    encoder = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),


    )

    
    front = nn.Sequential(
      nn.Upsample(scale_factor=2, mode='nearest'),
      nn.ReflectionPad2d((1, 1, 1, 1)),
      nn.Conv2d(512, 512, (3, 3)),
      nn.ReLU(),
      nn.Flatten(),

      nn.Linear(131072, 2048),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(2048, 2048),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(2048, 100)
    )
    

class classifier(nn.Module):
    def __init__(self,encoder,front=None):
        super(classifier,self).__init__()
        self.encoder=encoder

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.front = front
        if self.front == None:
            print("random")
            self.front = encoder_front.front
            for param in self.front.parameters():
                nn.init.normal_(param, mean=0.0, std=0.01)


    def encode(self,X):
        return(self.encoder(X))

    def classify(self,X):
        return(self.front(X))

    def forward(self,image):
        if self.training:
            encoded=self.encode(image)
            classes=self.classify(encoded)
            return classes

        else:
            encoded=self.encode(image)
            classes=self.classify(encoded)
            return classes
            print("no")

