import React from 'react';
import Card from 'react-bootstrap/Card';

import '../../node_modules/bootstrap/dist/css/bootstrap.min.css';

const ProcessedVideos = (props) => {

    var html = [];
    var idx = 0;
    for (; idx < props.title.length; idx++) {
        html.push(
        <div id={idx} style={{padding:'7%'}}>
            <Card>
                <Card.Header>{props.title[idx]}</Card.Header>
                <Card.Body>
                    <blockquote className="blockquote mb-0">
                    <p>
                        <a href={props.link[idx]}>{props.link[idx]}</a>
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