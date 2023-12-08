
with open("candieiro_armanda.JPG", "rb") as image_file:
    img = image_file.read()
    
for i in range(20, 40):
    with open(f"candieiro_armanda_{i}.jpg", "wb") as image_file:
        image_file.write(img)