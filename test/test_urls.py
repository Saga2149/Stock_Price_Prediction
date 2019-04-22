from django.test import SimpleTestCase
from django.urls import reverse,resolve
from stock.views import index,findList,user_login


class TestUrls(SimpleTestCase):
    def test_index(self):
        url = reverse('index')
        #print(resolve(url))
        self.assertEquals(resolve(url).func,index)

    def test_login(self):
        url = reverse('login')
        #print(resolve(url))
        self.assertEquals(resolve(url).func,user_login)        