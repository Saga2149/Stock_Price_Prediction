from django.test import TestCase,Client
from django.urls import reverse
from stock.models import Timestamp,UserProfileInfo
import json

class TestView(TestCase):
    def test_project(self):
        client = Client()
        response = client.get(reverse('index'))

        #self.assertEquals(response.status_code,200)

        #self.assertTemplateUsed(response,'charts.html') 