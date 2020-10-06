import React, {Component, useCallback, useMemo, useState} from 'react';

import Dropzone, {useDropzone} from 'react-dropzone';
import Button from 'react-bootstrap/Button';

/* CSS 3rd party imports */
import '../../../node_modules/bootstrap/dist/css/bootstrap.min.css';


const baseStyle = {
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  padding: '20px',
  marginLeft: '10%',
  marginRight: '13%',
  cursor: 'pointer',
  borderWidth: 2,
  borderRadius: 2,
  borderColor: '#eeeeee',
  borderStyle: 'dashed',
  backgroundColor: '#fafafa',
  color: '#bdbdbd',
  outline: 'none',
  transition: 'border .24s ease-in-out'
};

const activeStyle = {
  borderColor: '#2196f3'
};

const acceptStyle = {
  borderColor: '#00e676'
};

const rejectStyle = {
  borderColor: '#ff1744'
};

class MyDropzone extends Component{

  constructor() {
    super();
    this.state = {f1: null, f2: null};
  }

  on_drop1 = (acceptedFile) => {
    this.setState({f1: acceptedFile[0], f2: this.state.f2});
    console.log(this.state);
    document.getElementById('file1_upload').innerHTML = acceptedFile[0].path;
  }

  on_drop2 = (acceptedFile) => {
    this.setState({f1: this.state.f1, f2: acceptedFile[0]});
    console.log(this.state);
    document.getElementById('file2_upload').innerHTML = acceptedFile[0].path;
  }

  submit = function() {

    var null_files = false;
    if (this.state.f1 === null) {
      document.getElementById('file1_upload').innerHTML = 'You must submit an instructor video';
      null_files = true;
    }
    if (this.state.f2 === null) {
      document.getElementById('file2_upload').innerHTML = 'You must submit a student video';
      null_files = true;
    }

    if (!null_files) {

      let instructor = this.state.f1;
      let student    = this.state.f2;

      const form_data = new FormData();

      form_data.append('instructor', instructor);
      form_data.append('student', student);

      fetch('http://127.0.0.1:5000/upload_video', {
        method: 'POST',
        body: form_data,
      }).then(response => response.json())
        .then(response => console.log(response));
    }
  
  }

  render() {
    return (
      <div>
        <ol className='upload_instructions'>
            <li style={{paddingTop:'7%'}}>
              Drag and drop a video of your favourite instructor's class
              
              <Dropzone onDrop={this.on_drop1} accept='video/mp4' multiple={false}>
                {({getRootProps, getInputProps}) => (
                  <div {...getRootProps()} style={baseStyle}>
                    <input {...getInputProps()} />
                    Drop file here
                  </div>
                )}
              </Dropzone>
              <p id='file1_upload' />
            </li>
            <li>
              Drag and drop a video of a student following the instructor's class
              <Dropzone onDrop={this.on_drop2} accept='video/mp4' multiple={false}>
              {({getRootProps, getInputProps}) => (
                <div {...getRootProps()} style={baseStyle}>
                  <input {...getInputProps()} />
                  Drop file here
                </div>
              )}
            </Dropzone>
            <p id='file2_upload' />
            </li>
            <li>
              Click submit button and we will then process your video
              <div>
                <Button id='submit_button' size='lg' onClick={this.submit.bind(this)}>
                  Submit
                </Button>
                <p id='submit_message' style={{paddingTop:'3%', color: 'red'}} />
              </div>
                
            </li>
        </ol>
      </div>
    );
  }
}

export default MyDropzone;