from simple_image_download import simple_image_download as simp

response = simp.simple_image_download()
response.download(keywords="nature", limit=500, extensions={'.jpg', '.png', '.jpeg'})