# React Components

## What are Components?
Components are the building blocks of a React application. They can be functional or class-based.

### Functional Component Example
```jsx
function Welcome(props) {
  return <h1>Hello, {props.name}!</h1>;
}
```

### Class Component Example
```jsx
class Welcome extends React.Component {
  render() {
    return <h1>Hello, {this.props.name}!</h1>;
  }
}
```
