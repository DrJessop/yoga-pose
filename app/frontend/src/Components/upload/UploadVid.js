import React, {useCallback} from 'react';

import {useDropzone} from 'react-dropzone';

function MyDropzone() {
  const onDrop = useCallback(acceptedFiles => {
    
  }, [])
  const {getRootProps, getInputProps, isDragActive} = useDropzone({onDrop})

  return (
    <div className='upload' {...getRootProps()}>
      <input {...getInputProps()} />
      {
        isDragActive ?
          <center><p>Drop the files here ...</p></center> :
          <center><p>Drag 'n' drop some files here, or click to select files</p></center>
      }
    </div>
  )
}

export default MyDropzone;