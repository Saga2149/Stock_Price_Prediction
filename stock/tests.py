from django.test import TestCase
from django.contrib.auth.models import User

from stock.models import UserProfileInfo
# Create your tests here.

#Test Case for model
class Test1(TestCase):
	def setUp(self):
		#Job.objects.create(job_desc='SDE1')
		User.objects.create_user(username="test11", email="test@gmail.com", password="12345678")
		user=User.objects.get(username='test11')
		#job=Job.objects.get(job_desc='SDE1')
		UserProfileInfo.objects.create(user=user)

	def test1(self):
		#job=Job.objects.get(job_desc='SDE1')
		#self.assertTrue(isinstance(job,Job))
		us=User.objects.get(username='test11')
		g=UserProfileInfo.objects.get(user=us)
		self.assertTrue(isinstance(g,UserProfileInfo))