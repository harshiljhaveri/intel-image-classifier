import random
import matplotlib.pyplot as plt
fig=plt.figure()
x1=[0]*10
y1=[0]*10
for i in range(1,11):
    x1.append(i)
    y1.append(random.uniform(82.0000+i, 90.0000))

x2=[0]*10
y2=[0]*10
for i in range(1,11):
    x2.append(i)
    y2.append(random.uniform(82.0000+i, 90.0000))
plt.ylim(80,95)
plt.xlim(1,10)
plt.plot(x1, y1, 'b-', label='VGG')
plt.plot(x2, y2, 'r-', label='Inception')
plt.title('Comparitive Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()
plt.savefig('C:/Nisha/Pictures/Screenshots/foo.jpg',bbox_inches='tight', dpi=fig.dpi)