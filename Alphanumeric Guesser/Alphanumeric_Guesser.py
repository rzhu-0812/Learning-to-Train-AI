from keras import models
from PIL import Image, ImageDraw
import customtkinter as ctk
import numpy as np

model = models.load_model('emnistModel.keras')
label_to_char = {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8", "9": "9", "10": "A", "11": "B", "12": "C", "13": "D", "14": "E", "15": "F", "16": "G", "17": "H", "18": "I", "19": "J", "20": "K", "21": "L", "22": "M", "23": "N", "24": "O", "25": "P", "26": "Q", "27": "R", "28": "S", "29": "T", "30": "U", "31": "V", "32": "W", "33": "X", "34": "Y", "35": "Z", "36": "a", "37": "b", "38": "c", "39": "d", "40": "e", "41": "f", "42": "g", "43": "h", "44": "i", "45": "j", "46": "k", "47": "l", "48": "m", "49": "n", "50": "o", "51": "p", "52": "q", "53": "r", "54": "s", "55": "t", "56": "u", "57": "v", "58": "w", "59": "x", "60": "y", "61": "z"}

def correct_orientation(image_array):
  image_array = image_array.reshape(28, 28)
  corrected = np.fliplr(np.rot90(image_array, k=-1))
  return corrected.reshape(1, 28, 28, 1)

def center_image(img):
    img_array = np.array(img)
    if np.sum(img_array) == 0:
        return img

    non_zero_coords = np.argwhere(img_array > 0)
    top, left = non_zero_coords.min(axis=0)
    bottom, right = non_zero_coords.max(axis=0)

    cropped = img.crop((left, top, right, bottom))
    new_img = Image.new('L', (28, 28), 0)

    new_width, new_height = cropped.size
    paste_x = (28 - new_width) // 2
    paste_y = (28 - new_height) // 2

    new_img.paste(cropped, (paste_x, paste_y))
    return new_img

class GuesserGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Alphanumeric Recognizer")
        self.geometry("420x550")

        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        self.mainFrame = ctk.CTkFrame(self, corner_radius=15)
        self.mainFrame.pack(padx=25, pady=25, fill="both", expand=True)

        self.titleLabel = ctk.CTkLabel(
            self.mainFrame,
            text="Alphanumeric Recognizer",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        self.titleLabel.pack(pady=(15, 25))

        self.canvasFrame = ctk.CTkFrame(self.mainFrame, fg_color="#1A1A1A", corner_radius=15)
        self.canvasFrame.pack(padx=25, pady=15)

        self.canvas = ctk.CTkCanvas(self.canvasFrame, width=300, height=300, bg='white', highlightthickness=0)
        self.canvas.pack(padx=4, pady=4)

        self.result = ctk.CTkLabel(
            self.mainFrame,
            text="--",
            font=ctk.CTkFont(size=48, weight="bold")
        )
        self.result.pack(pady=15)

        self.buttonFrame = ctk.CTkFrame(self.mainFrame, fg_color="transparent")
        self.buttonFrame.pack(pady=15, fill="x")

        self.clearBtn = ctk.CTkButton(
            self.buttonFrame,
            text='Clear',
            command=self.clearCanvas,
            width=130,
            height=45,
            font=ctk.CTkFont(size=18),
            corner_radius=12,
            hover_color="#B22222",
            fg_color="#444444",
            border_width=1,
            border_color="#666666"
        )
        self.clearBtn.pack(side="left", padx=25, pady=10, expand=True)

        self.predictBtn = ctk.CTkButton(
            self.buttonFrame,
            text='Predict',
            command=self.predictChar,
            width=130,
            height=45,
            font=ctk.CTkFont(size=18),
            corner_radius=12,
            hover_color="#006400",
            fg_color="#0066CC"
        )
        self.predictBtn.pack(side="right", padx=25, pady=10, expand=True)

        self.image = Image.new('L', (28, 28), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind('<B1-Motion>', self.drawCanvas)
        self.lastX, self.lastY = None, None
        self.canvas.bind('<ButtonRelease-1>', self.resetPosition)

    def resetPosition(self, event):
        self.lastX, self.lastY = None, None

    def drawCanvas(self, event):
        x, y = event.x, event.y
        brushSize = 18

        if self.lastX and self.lastY:
            self.canvas.create_line(self.lastX, self.lastY, x, y, width=brushSize, fill='black', capstyle='round', smooth=True)
            scaledX, scaledY = x / (300/28), y / (300/28)
            lastScaledX, lastScaledY = self.lastX / (300/28), self.lastY / (300/28)
            self.draw.line([lastScaledX, lastScaledY, scaledX, scaledY], fill=255, width=2)
        else:
            self.canvas.create_oval(
                x - brushSize / 2,
                y - brushSize / 2,
                x + brushSize / 2,
                y + brushSize / 2,
                fill='black',
                outline='black'
            )
            scaledX, scaledY = x / (300/28), y / (300/28)
            self.draw.ellipse([
                scaledX - 0.8,
                scaledY - 0.8,
                scaledX + 0.8,
                scaledY + 0.8
            ], fill=255)

        self.lastX, self.lastY = x, y

    def clearCanvas(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (28, 28), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result.configure(text="--")

    def predictChar(self):
        centered_img = center_image(self.image)

        img_arr = np.array(centered_img) / 255.0
        img = img_arr.reshape(1, 28, 28, 1)
        
        if np.sum(img) == 0:
            self.result.configure(text="--")
            return
            
        img = correct_orientation(img)

        pred = model.predict(img, verbose=0)
        predictedIdx = np.argmax(pred)
        confidence = np.max(pred) * 100

        print(f"Predicted index: {predictedIdx}, confidence: {confidence:.1f}")

        predictedChar = label_to_char[str(predictedIdx)]
        self.result.configure(text=f"{predictedChar}")

if __name__ == '__main__':
    app = GuesserGUI()
    app.mainloop()
