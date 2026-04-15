from typing import Annotated, Literal

from pydantic import BaseModel, Field


class MakeDirectoryAction(BaseModel):
    action_type: Literal["make_directory"] = "make_directory"
    path: str = Field(min_length=1)


class CreateFileAction(BaseModel):
    action_type: Literal["create_file"] = "create_file"
    path: str = Field(min_length=1)
    content: str = ""
    content_inferred: bool = False


class ReadFileAction(BaseModel):
    action_type: Literal["read_file"] = "read_file"
    path: str = Field(min_length=1)


class DeletePathAction(BaseModel):
    action_type: Literal["delete_path"] = "delete_path"
    path: str = Field(min_length=1)


class RequestExecutionAction(BaseModel):
    action_type: Literal["request_execution"] = "request_execution"
    target: str = Field(min_length=1)


class ListFilesAction(BaseModel):
    action_type: Literal["list_files"] = "list_files"
    path: str = "."


CodingAction = Annotated[
    MakeDirectoryAction
    | CreateFileAction
    | ReadFileAction
    | DeletePathAction
    | RequestExecutionAction
    | ListFilesAction,
    Field(discriminator="action_type"),
]


class CodingActionBatch(BaseModel):
    actions: list[CodingAction] = Field(default_factory=list)


class CodingActionResult(BaseModel):
    action_type: str
    target: str | None = None
    status: Literal["completed", "blocked", "error", "approval_required"]
    message: str
    details: dict = Field(default_factory=dict)
