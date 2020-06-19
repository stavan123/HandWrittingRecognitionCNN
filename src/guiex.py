from tkinter import *    
from main import main  
  
def opencam():
    main()
# create a tkinter window 
root = Tk()                 
# Open window having dimension 100x100 
root.geometry('200x200')
    

	
# Create a Button 
btn = Button(root, text = 'Open Camera', bd = '5', 
                          command = opencam)  
  
# Set the position of button on the top of window.    
btn.pack(side = 'top')     
  
root.mainloop() 

