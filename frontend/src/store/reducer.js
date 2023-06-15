import { combineReducers } from "redux";
import imageReducer from "./imageReducer";
import optionReducer from "./optionReducer";
import reportReducer from "./reportReducer";
import resReducer from "./resReducer";

const reducer = combineReducers({
    image: imageReducer,
    res: resReducer,
    option: optionReducer,
    report: reportReducer
});

export default reducer; 