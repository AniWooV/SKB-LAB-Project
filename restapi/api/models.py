from django.db import models


class Analysis(models.Model):
    patient_name = models.CharField(verbose_name='Patient name', max_length=128, default='Unknown patient')
    analysis = models.FileField(verbose_name='File', upload_to='analysis/', default='analysis/none/nothing.pdf')
    date_uploaded = models.DateTimeField(verbose_name='Download date', auto_now_add=True)
    processing_result = models.TextField(verbose_name='Result', editable=False, default='None')
    processing_completed = models.BooleanField(verbose_name='Is completed', editable=False, default=False)

    def __str__(self):
        return self.analysis.path
