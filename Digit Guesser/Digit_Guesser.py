from keras import models
from PIL import Image, ImageDraw
import customtkinter as ctk
import numpy as np

model = models.load_model('mnistModel.keras')

class GuesserGUI(ctk.CTk):
  def __init__(self):
    super().__init__()
    self.title("Digit Recognizer")
    self.geometry("420x550")
    
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")
    
    self.mainFrame = ctk.CTkFrame(self, corner_radius=15)
    self.mainFrame.pack(padx=25, pady=25, fill="both", expand=True)

    self.titleLabel = ctk.CTkLabel(
      self.mainFrame, 
      text="Digit Recognizer", 
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
      command=self.predictDig,
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
      self.draw.line([lastScaledX, lastScaledY, scaledX, scaledY], fill=255, width=1)
      
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
  
  def predictDig(self):
    self.update()
    
    img = (np.array(self.image.resize((28, 28)).convert('L')) / 255.0).reshape(1, 28, 28, 1)
    
    if np.sum(img) == 0:
      self.result.configure(text="--")
      return
    
    pred = model.predict(img, verbose=0)
    predictedDig = np.argmax(pred)

    self.result.configure(text=f"{predictedDig}")

if __name__ == '__main__':
  app = GuesserGUI()
  app.mainloop()