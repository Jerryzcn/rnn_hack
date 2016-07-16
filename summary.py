
class Summary:
    def __init__(self, name):
        self.name = name

    def start(self):
        self.f=open('./log/'+self.name+'.txt','w')
        self.f.close()


    def log(self, val):
        self.f=open('./log/'+self.name+'.txt','a')
        self.f.write(str(val)+'\n')
        self.f.close()
