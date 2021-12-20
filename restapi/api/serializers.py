import traceback

from threading import Thread
from rest_framework import serializers
from rest_framework.serializers import raise_errors_on_nested_writes
from rest_framework.utils import model_meta

from .models import *


from parsers .main_parsers import *


class AnalysisDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = Analysis
        fields = ('id', 'patient_name', 'date_uploaded', 'analysis', 'processing_completed', 'processing_result')

    def create(self, validated_data):
        raise_errors_on_nested_writes('create', self, validated_data)

        ModelClass = self.Meta.model

        info = model_meta.get_field_info(Analysis)
        many_to_many = {}
        for field_name, relation_info in info.relations.items():
            if relation_info.to_many and (field_name in validated_data):
                many_to_many[field_name] = validated_data.pop(field_name)

        try:
            instance = ModelClass._default_manager.create(**validated_data)
        except TypeError:
            tb = traceback.format_exc()
            msg = (
                'Got a `TypeError` when calling `%s.%s.create()`. '
                'This may be because you have a writable field on the '
                'serializer class that is not a valid argument to '
                '`%s.%s.create()`. You may need to make the field '
                'read-only, or override the %s.create() method to handle '
                'this correctly.\nOriginal exception was:\n %s' %
                (
                    ModelClass.__name__,
                    ModelClass._default_manager.name,
                    ModelClass.__name__,
                    ModelClass._default_manager.name,
                    self.__class__.__name__,
                    tb
                )
            )
            raise TypeError(msg)

        # Save many-to-many relationships after the instance is created.
        if many_to_many:
            for field_name, value in many_to_many.items():
                field = getattr(instance, field_name)
                field.set(value)

        Thread(target=parse_analysis, args=(instance, )).start()

        return instance


def parse_analysis(analysis):
    path = analysis.analysis.path.replace('\\', '\\\\')

    analysis.processing_result = str(get_parsed_analysis_with_camelot(path))
    analysis.processing_completed = True

    analysis.save()


class AnalysisListSerializer(serializers.ModelSerializer):
    class Meta:
        model = Analysis
        fields = ('id', 'patient_name', 'date_uploaded', 'processing_completed')
