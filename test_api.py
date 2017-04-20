import base64
import urllib
import urllib2

# model = 'starry_night'
model = 'mosaic'

f = open('./test_img/tree.jpg','rb')
img = base64.b64encode(f.read())

test_data = {'image':img}
test_data_urlencode = urllib.urlencode(test_data)
requrl = 'http://127.0.0.1:5000/image_stylization/'+model+'/'
req = urllib2.Request(url=requrl, data=test_data_urlencode)

# req.add_header("Content-Type", "application/x-www-form-urlencoded")

res_data = urllib2.urlopen(req)
res = res_data.read()

# print res

file = 'out.jpg'
with open(file, 'wb') as f:
  f.write(res)