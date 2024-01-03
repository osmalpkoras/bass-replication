# Paper Title: A Second Look on BASS - Boosting Abstractive Summarization with Unified Semantic Graphs
# Author: Osman Alperen Koras
# Affiliation: Institute for AI in Medicine (IKIM), University Medicine Essen
# Email: osman.koras@uni-due.de
# Date: 12.10.2023
#
# License: MIT
import os
from typing import IO, Any

import javaobj.v2 as javaobj

from javaobj.v2.api import ObjectTransformer
from javaobj.v2.transformers import DefaultObjectTransformer, NumpyArrayTransformer
from javaobj.utils import java_data_fd

from javaobj.constants import (
    StreamConstants,
    TerminalCode,
    TypeCode,
    PRIMITIVE_TYPES,
)

from javaobj.v2.beans import (
    ParsedJavaContent,
    BlockData,
    JavaClassDesc,
    JavaClass,
    JavaArray,
    JavaEnum,
    JavaField,
    JavaInstance,
    JavaString,
    ExceptionState,
    ExceptionRead,
    ClassDescType,
    FieldType,
    ClassDataType,
)


class JavaObjectStreamReader:
    def __init__(self):
        # if a different version is included, we have to check if our code, which is mostly a copy
        # of the code given by the original library, is still up to date/in alignment with the library code
        assert javaobj.__version_info__ == (0, 4, 3)
        self._first_load = True
        self.parser: JavaStreamParser = None

    # a load function that behaves more like the pickle.load method
    def load(self, file_object, *transformers, **kwargs):
        # type: (IO[bytes], ObjectTransformer, Any) -> Any
        """
        Deserializes Java primitive data and objects serialized using
        ObjectOutputStream from a file-like object.

        :param file_object: A file-like object
        :param transformers: Custom transformers to use
        :return: The deserialized object
        """
        if self._first_load:
            # Check file format (uncompress if necessary)
            file_object = java_data_fd(file_object)

            # Ensure we have the default object transformer
            all_transformers = list(transformers)
            for t in all_transformers:
                if isinstance(t, DefaultObjectTransformer):
                    break
            else:
                all_transformers.append(DefaultObjectTransformer())

            if kwargs.get("use_numpy_arrays", False):
                # Use the numpy array transformer if requested
                all_transformers.append(NumpyArrayTransformer())

            # Parse the object(s)
            self.parser = JavaStreamParser(file_object, all_transformers)
            self._first_load = False

        try:
            content = self.parser.run()
        except EOFError:
            raise EOFError

        return content


class JavaStreamParser(javaobj.core.JavaStreamParser):
    def __init__(self, fd, transformers):
        super().__init__(fd, transformers)
        self._first_run = True


    def run(self):
        # type: () -> ParsedJavaContent
        """
        Parses the input stream
        """

        if self._first_run:
            # Check the magic byte
            magic = self.__reader.read_ushort()
            if magic != StreamConstants.STREAM_MAGIC:
                raise ValueError("Invalid file magic: 0x{0:x}".format(magic))

            # Check the stream version
            version = self.__reader.read_ushort()
            if version != StreamConstants.STREAM_VERSION:
                raise ValueError("Invalid file version: 0x{0:x}".format(version))

            # Reset internal state
            self._reset()

            # Read content
            contents = []  # type: List[ParsedJavaContent]
            self._first_run = False

        c: ParsedJavaContent
        break_due_eoferror = False
        while True:
            self._log.info("Reading next content")
            start = self.__fd.tell()
            try:
                type_code = self.__reader.read_byte()
            except EOFError:
                break_due_eoferror = True
                # End of file
                break

            if type_code == TerminalCode.TC_RESET:
                # Explicit reset
                self._reset()
                continue

            parsed_content = self._read_content(type_code, True)
            self._log.debug("Read: %s", parsed_content)
            if parsed_content is not None and parsed_content.is_exception:
                # Get the raw data between the start of the object and our
                # current position
                end = self.__fd.tell()
                self.__fd.seek(start, os.SEEK_SET)
                stream_data = self.__fd.read(end - start)

                # Prepare an exception object
                parsed_content = ExceptionState(parsed_content, stream_data)

            c = parsed_content
            break

        if break_due_eoferror:
            for content in self.__handles.values():
                content.validate()

            # TODO: connect member classes ? (see jdeserialize @ 864)

            # if self.__handles:
            #     self.__handle_maps.append(self.__handles.copy())

            # we have reached the end of this file
            raise EOFError

        return c

    def _reset(self):

        """
        Resets the internal state of the parser
        """
        # if self.__handles:
        #     self.__handle_maps.append(self.__handles.copy())

        self.__handles.clear()

        # Reset handle index
        self.__current_handle = StreamConstants.BASE_REFERENCE_IDX

