import base64
import urllib
import urllib2

f = open('./images/cat11.jpg','rb')
m1 = base64.b64encode(f.read())

test_data = {'m1':m1, 'm2':m2}
test_data_urlencode = urllib.urlencode(test_data)
requrl = 'http://127.0.0.1:5001/image_match_api/'
req = urllib2.Request(url=requrl, data=test_data_urlencode)

# req.add_header("Content-Type", "application/x-www-form-urlencoded")

res_data = urllib2.urlopen(req)
res = res_data.read()

print res