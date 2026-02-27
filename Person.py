
class Person:
    def __init__(self):
        self.state="SITTING"
        self.knee_angle=0
    
    def update(self, angle, movement):
       self.knee_angle=angle
       if(movement>10):
           self.state="WALKING"
       elif(self.knee_angle>130):
           self.state="STANDING"
       else : self.state="SITTING"
       return self.state


