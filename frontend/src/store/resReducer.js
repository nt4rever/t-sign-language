import { RES_CHANGE } from './actions';

const initialState = {
    image: "",
};

const resReducer = (state = initialState, action) => {
    switch (action.type) {
        case RES_CHANGE:
            return {
                ...state,
                res: action.res
            }
        default:
            return state;
    }
}

export default resReducer;