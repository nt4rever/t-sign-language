import * as React from 'react';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import Box from '@mui/material/Box';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import CloseIcon from '@mui/icons-material/Close';
import Slide from '@mui/material/Slide';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import { useDispatch, useSelector } from 'react-redux';
import { REPORT_CHANGE } from "../../store/actions";

const Transition = React.forwardRef(function Transition(props, ref) {
    return <Slide direction="up" ref={ref} {...props} />;
});

export default function Report() {

    const { open, data, cm } = useSelector((state) => state.report)
    const dispatch = useDispatch()
    const handleClose = () => {
        dispatch({
            type: REPORT_CHANGE,
            open: false
        })
    };

    return (
        <div>
            <Dialog
                fullScreen
                open={open}
                onClose={handleClose}
                TransitionComponent={Transition}
            >
                <AppBar sx={{ position: 'relative' }}>
                    <Toolbar>
                        <IconButton
                            edge="start"
                            color="inherit"
                            onClick={handleClose}
                            aria-label="close"
                        >
                            <CloseIcon />
                        </IconButton>
                        <Typography sx={{ ml: 2, flex: 1 }} variant="h6" component="div">
                            Report
                        </Typography>
                        <Button autoFocus color="inherit" onClick={handleClose}>
                            close
                        </Button>
                    </Toolbar>
                </AppBar>

                <Box sx={{ width: '100%', p: 0, display: 'flex', justifyContent: 'center' }}>
                    <Box sx={{ width: '60%', p: 5 }}>
                        <TableContainer component={Paper}>
                            <Table sx={{ minWidth: 650 }} aria-label="simple table">
                                <TableHead>
                                    <TableRow>
                                        <TableCell>*</TableCell>
                                        <TableCell align="right">precision</TableCell>
                                        <TableCell align="right">recall</TableCell>
                                        <TableCell align="right">f1-score</TableCell>
                                        <TableCell align="right">support</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {data && Object.keys(data).map((keyName, i) => {
                                        if (keyName == "accuracy") {
                                            return (<TableRow
                                                key={i}
                                                sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
                                            >
                                                <TableCell component="th" scope="row">
                                                    {keyName}
                                                </TableCell>
                                                <TableCell align="right">{Math.round(data[keyName] * 100) / 100}</TableCell>
                                            </TableRow>)
                                        } else {
                                            return (
                                                <TableRow
                                                    key={i}
                                                    sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
                                                >
                                                    <TableCell component="th" scope="row">
                                                        {keyName}
                                                    </TableCell>
                                                    <TableCell align="right">{Math.round(data[keyName].precision * 100) / 100}</TableCell>
                                                    <TableCell align="right">{Math.round(data[keyName].recall * 100) / 100}</TableCell>
                                                    <TableCell align="right">{Math.round(data[keyName]["f1-score"] * 100) / 100}</TableCell>
                                                    <TableCell align="right">{Math.round(data[keyName].support * 100) / 100}</TableCell>
                                                </TableRow>
                                            )
                                        }
                                    })}
                                </TableBody>
                            </Table>
                        </TableContainer>
                    </Box>
                </Box>
                <Box sx={{ width: '100%', p: 5, display: 'flex', justifyContent: 'center' }}>
                    <div>
                        {cm && (
                            <img src={"data:image/png;charset=utf-8;base64," + cm} width="100%" />
                        )}
                    </div>
                </Box>
            </Dialog>
        </div>
    );
}
