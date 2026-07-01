import dataclasses
import tempfile
from pathlib import Path

from rest_framework.response import Response
from rest_framework.views import APIView

from rag.service import get_rag_service

from .serializers import AskQuestionSerializer, DocumentUploadSerializer


class HealthView(APIView):
    def get(self, request):
        return Response(get_rag_service().health())


class AskQuestionView(APIView):
    def post(self, request):
        serializer = AskQuestionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        result = get_rag_service().ask(data["question"], data.get("history"))
        return Response(dataclasses.asdict(result))


class DocumentUploadView(APIView):
    def post(self, request):
        serializer = DocumentUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        upload = serializer.validated_data["file"]

        suffix = "." + upload.name.rsplit(".", 1)[-1].lower()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                for chunk in upload.chunks():
                    tmp.write(chunk)
                tmp_path = tmp.name

            result = get_rag_service().ingest_file(tmp_path)
            return Response(result)
        finally:
            # we only keep the chunks in ChromaDB, not the original file
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
