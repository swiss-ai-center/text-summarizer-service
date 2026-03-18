from common_code.config import get_settings
from common_code.logger.logger import get_logger, Logger
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from common_code.tasks.models import TaskData
# Imports required by the service's model
from transformers import pipeline

settings = get_settings()
classifier = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")

api_summary = """
Summarize the given text.
"""

api_description = """
Summarize the given text using the HuggingFace transformers library with model bart-large-cnn-samsum.
"""
api_title = "Text summarizer API."
version = "0.0.1"


class MyService(Service):
    """
    Text summarizer model
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="Text Summarizer",
            slug="text-summarizer",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(name="text", type=[FieldDescriptionType.TEXT_PLAIN]),
            ],
            data_out_fields=[
                FieldDescription(name="result", type=[FieldDescriptionType.TEXT_PLAIN]),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.NATURAL_LANGUAGE_PROCESSING,
                    acronym=ExecutionUnitTagAcronym.NATURAL_LANGUAGE_PROCESSING,
                ),
            ],
            has_ai=True,
            docs_url="https://docs.swiss-ai-center.ch/reference/services/text-summarizer/",
        )
        self._logger = get_logger(settings)

    def process(self, data):
        # Get the text to analyze from storage
        text = data["text"].data
        # Convert bytes to string
        text = text.decode("utf-8")

        # Limit the text to 142 words
        text = " ".join(text.split()[:500])

        # Run the model
        result = classifier(text, max_length=100, min_length=5, do_sample=False)

        # Convert the result to bytes
        file_bytes = result[0]["summary_text"].encode("utf-8")

        return {
            "result": TaskData(
                data=file_bytes,
                type=FieldDescriptionType.TEXT_PLAIN
            )
        }
