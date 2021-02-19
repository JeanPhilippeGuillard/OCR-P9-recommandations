# Bookshelf

Bookshelf is a react-native app used in the project #9 of the AI Engineer Open Classrooms Path.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Setup](#setup)
- [Sharing](#sharing)

## Requirements

- [Node.js](https://nodejs.org/)
- [npm](https://www.npmjs.com/get-npm)

if you want to preview the app on a real phone, you need to install the [android](https://play.google.com/store/apps/details?id=host.exp.exponent) or [iOS](https://apps.apple.com/app/expo-client/id982107779) expo app

_Expo is platform to preview mobile apps. It uses either the iOS or android simulator, depending on the type on machine it's running on. It's also possible to download the expo app from either the app store or the play store to preview mobile apps directly from a mobile phone_

## Installation

First, make sure you have [node](https://nodejs.org/) and [npm](https://www.npmjs.com/get-npm) installed.

Then, clone this repo, install required dependencies using `npm install` in the project directory.

To start the app, execute `npm start`, and scan the provided QR Code with a phone to preview the app.

## Setup

Once a user is selected within the app, recommendations for this user will be fetched automatically.

Precisely, an HTTP POST request will be send to the address defined by the variable `API_URL`, which is configured in `config.json`. A json body having this shape is expected:

```
{
   userId: __SOME_USER_ID__
}
```

And a response whose body is a valid json array containing the ids of the suggested books will be returned by the API, for example:

```
[34, 32, 893, 1]
```

_Tip : To setup the API, we suggest using [Azure Functions](https://docs.microsoft.com/fr-fr/azure/azure-functions/functions-create-first-function-python). Once the http endpoint provided by azure has been set in `config.json`, the userId provided can be retrieved using `req.get_json()`_

## Sharing

This project is ready to be shared with [Expo](https://expo.io/). Head to expo documentation to learn how to share a preview of your app.
