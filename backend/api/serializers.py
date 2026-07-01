from rest_framework import serializers

from rag.pipeline.document_processor import DocumentProcessor


class DocumentUploadSerializer(serializers.Serializer):
    file = serializers.FileField()

    def validate_file(self, value):
        ext = "." + value.name.rsplit(".", 1)[-1].lower()
        if ext not in DocumentProcessor.supported_extensions():
            raise serializers.ValidationError(f"Unsupported file type '{ext}'.")
        return value


class ChatMessageSerializer(serializers.Serializer):
    role = serializers.ChoiceField(choices=["user", "assistant"])
    text = serializers.CharField(max_length=4000)


class AskQuestionSerializer(serializers.Serializer):
    question = serializers.CharField(max_length=2000)
    history = ChatMessageSerializer(many=True, required=False, default=list)
