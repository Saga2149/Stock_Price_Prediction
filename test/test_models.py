from django.test import TestCase
from stock.models import Timestamp,UserProfileInfo


class TestModels(TestCase):
    
    def setUp(self):
    #     u = UserProfileInfo()
    #     userinfo= u.user
    #     self.user = UserProfileInfo.objects.create(
    #         userinfo = 'ravi',
    #     ) 

    # def testModel(self):
    #     self.assertEquals(self.user.slug,'user')    