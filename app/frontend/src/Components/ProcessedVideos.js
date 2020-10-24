import React from 'react';
import Card from 'react-bootstrap/Card';

import '../../node_modules/bootstrap/dist/css/bootstrap.min.css';

const ProcessedVideos = (props) => {

    var html = [];
    console.log(props);
    for (var idx = 0; idx < props.title.length; idx++) {
        html.push(
        <div id={idx} style={{padding:'7%'}}>
            <Card>
                <Card.Header>{props.title[idx]}</Card.Header>
                <Card.Body>
                    <blockquote className="blockquote mb-0">
                    <p>
                    <video muted controls className='videos'>
                        <source src={props.path[idx]} type='video/mp4' />
                    </video>
                    </p>
                    <footer className="blockquote-footer">
                        {props.date[idx]}
                    </footer>
                    </blockquote>
                </Card.Body>
            </Card>
        </div>);
    }

    return (
        <div>
            {html}
        </div>
    );
}

export default ProcessedVideos;